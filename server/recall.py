import argparse
import importlib

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger
from server.models.base import RecallModel, RecallModel_w_CTR
from server.utils import parse_command_line_args

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="AddRet")
    parser.add_argument("--model_config_file_path", type=str, default="./server/config/AddRet.yaml")
    parser.add_argument("--ctr_model_config_file_path", type=str, default=None)

    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--port", type=int, default=6001, help="Port number for the server")


    return parser.parse_known_args()

args, unparsed_args = parse_args()
command_line_configs = parse_command_line_args(unparsed_args)


with open(args.model_config_file_path) as f:
    model_config = yaml.safe_load(f)

ctr_config = None
if args.ctr_model_config_file_path != None:
    with open(args.ctr_model_config_file_path) as f:
        ctr_config = yaml.safe_load(f)

model_config.update(command_line_configs)
print("model_config:", model_config)
if ctr_config == None:
    recall_model = RecallModel(args.model_name, model_config)
else:
    recall_model = RecallModel_w_CTR(args.model_name, model_config, ctr_config)
app = FastAPI()


@app.post("/recall")
async def recall(request: Request):
    data = await request.json()

    items_list = recall_model.recall(**data)
    result = {"item_list": items_list}
    logger.info(f"Sent JSON: {result}")
    return JSONResponse(result)


uvicorn.run(app, host=args.host, port=args.port, log_level="info")
