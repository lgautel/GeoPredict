import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.99, device=None, zero_stage=3):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
        
        for buffer, buffer_ema in zip(model.buffers(),
                                      model_ema.buffers()):
            data = buffer.data
            if device is not None:
                data = data.to(device)
            
            if data.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                buffer_ema.data.copy_(torch.lerp(data, buffer_ema.data, beta))
            else:
                buffer_ema.data.copy_(data)


def save_zero_three_model(model_ema, global_rank, output_model_file, zero_stage=3):
    zero_stage_3 = (zero_stage == 3)
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        
        for k, v in model_to_save.named_buffers():
            v_b = v.cpu()
            if global_rank == 0:
                output_state_dict[k] = v_b
        
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
