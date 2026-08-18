#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=18, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )
    # --- additions ---
    parser.add_argument(
        "--dynamo", action="store_true",
        help="force the torch.export-based exporter (requires opset >= 18)"
    )
    parser.add_argument(
        "--external-data", action="store_true",
        help="store weights in a separate .onnx.data file instead of inlining them"
    )
    parser.add_argument(
        "--max-size", type=int, default=2048,
        help="upper bound for dynamic height/width, must be a multiple of 32"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    # The dynamo exporter has no translations below opset 18; asking for less makes it
    # export at 18 and then fail the down-conversion (no Resize adapter). Only use it
    # when the requested opset actually allows it.
    use_dynamo = args.dynamo or args.opset >= 18
    if args.dynamo and args.opset < 18:
        logger.warning("--dynamo with opset {} will emit opset 18".format(args.opset))

    export_kwargs = {"dynamo": use_dynamo}
    if use_dynamo:
        # external_data defaults to True since torch 2.9, which splits the weights into
        # a sidecar .onnx.data file. YOLOX is far below the 2 GB protobuf limit, so keep
        # everything in one file unless asked otherwise.
        export_kwargs["external_data"] = args.external_data
        if args.dynamic:
            # dynamo ignores dynamic_axes; shapes are declared as torch.export Dims.
            # Given as a tuple, they are matched positionally against forward()'s args.
            export_kwargs["dynamic_shapes"] = ({
                0: torch.export.Dim("batch", min=1),
                2: torch.export.Dim("height", min=32, max=args.max_size),
                3: torch.export.Dim("width", min=32, max=args.max_size),
            },)
    elif args.dynamic:
        export_kwargs["dynamic_axes"] = {
            args.input: {0: "batch", 2: "height", 3: "width"},
            args.output: {0: "batch"},
        }

    torch.onnx.export(
        model,
        (dummy_input,),
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        opset_version=args.opset,
        **export_kwargs,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        onnx_model = onnx.load(args.output_name)
        try:
            import onnxslim
            onnx_model = onnxslim.slim(onnx_model)
        except ImportError:
            from onnxsim import simplify
            onnx_model, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
        # onnx.load pulls external tensors back in and clears their EXTERNAL location,
        # so this save is self-contained regardless of how the export was written.
        onnx.save(onnx_model, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    import onnx
    final = onnx.load(args.output_name)
    onnx.checker.check_model(final)
    shape = [d.dim_param or d.dim_value for d in final.graph.input[0].type.tensor_type.shape.dim]
    logger.info("opset {}, ir_version {}, input shape {}".format(
        final.opset_import[0].version, final.ir_version, shape))


if __name__ == "__main__":
    main()
