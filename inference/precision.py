import torch

def _keep_selected_tensors_fp32(model, keep_in_fp32_modules):
    """
    Keep selected parameters/buffers in FP32 based on name tokens.

    Diffusers-style `_keep_in_fp32_modules` is typically a list of name fragments
    (e.g. "time_embedder", "norm1", "scale_shift_table").
    """
    if not keep_in_fp32_modules:
        return

    keep_tokens = tuple(keep_in_fp32_modules)

    for name, param in model.named_parameters(recurse=True):
        if torch.is_floating_point(param) and any(token in name for token in keep_tokens):
            param.data = param.data.to(torch.float32)

    for name, buf in model.named_buffers(recurse=True):
        if torch.is_floating_point(buf) and any(token in name for token in keep_tokens):
            buf.data = buf.data.to(torch.float32)
