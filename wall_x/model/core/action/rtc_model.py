"""Compatibility entry point; Qwen private context/cache live in its adapter."""


def generate_rtc_action(model, ctx, condition):
    from wall_x.model.adapters.qwen_flow import generate_qwen_flow

    return generate_qwen_flow(model, ctx, condition)
