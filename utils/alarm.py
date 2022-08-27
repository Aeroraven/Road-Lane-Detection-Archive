import os
import sys
import warnings


def show_warning(text):
    warnings.warn("[Project Warning]" + text, UserWarning)


def alarm_confirm(st, fatal=False):
    #show_warning(st)
    print("[Warning]",st)
    if fatal:
        print("Please solve the warning above before running the new instance. Check global_config_server_side.yaml or global_config_client_side.yaml before rerun the script.")
        raise Exception("Program terminated at the parameter checking procedure")
    print("Enter [Y/y] to confirm, or exit the program otherwise.", file=sys.stderr)
    x = input()
    print("Your input:",x)
    if x == "y" or x == "Y":
        return
    raise Exception("Program terminated at the parameter checking procedure. Check global_config_server_side.yaml or global_config_client_side.yaml before rerun the script.")


def config_pre_check(**kwargs):
    # Device Check
    if kwargs['device'] == 'cpu':
        alarm_confirm("[Device]: Are you sure to continue training without CUDA?")
    if kwargs['num_workers'] == 0:
        alarm_confirm("[Device]: Are you sure to continue without any dataloader worker?")
    # Path Check
    path_list = ['train_image_path',
                 'test_image_path',
                 'base_workspace_path'
                 ]
    for i in path_list:
        if not os.path.exists(kwargs[i]):
            alarm_confirm("[Path Inaccessible]: YAML config item "
                          "" + str(i) + " path: " + str(kwargs[i]) + " is not accessible.", fatal=True)
    pass
