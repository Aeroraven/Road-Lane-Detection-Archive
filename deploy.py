import torch
import torchsummary

from utils.utility import get_config_json, get_model


def deploy(**kwargs):
    print("Deployment starts")
    fargs = {}
    for key in kwargs.keys():
        fargs[key] = kwargs[key]
    input_layer_name = ['input']
    output_layer_name = ['output']
    print("Loading runner")
    torch_model = get_model(mode="deploy", **fargs).to(fargs["device"])
    torch_model.load_state_dict(torch.load(fargs['input_model_path']))
    input_var = torch.autograd.Variable(torch.randn((1, 3, fargs['image_scale_h'], fargs['image_scale_w'])))
    if fargs['deploy_device'] == 'cuda':
        input_var = input_var.cuda()
    torch.onnx.export(torch_model, input_var, fargs['output_model_path'],
                      input_names=input_layer_name, output_names=output_layer_name,
                      verbose=True, opset_version=13, do_constant_folding=True)


if __name__ == "__main__":
    deploy(**get_config_json())
