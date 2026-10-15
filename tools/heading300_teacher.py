"""Full deterministic frozen teacher action, including the learned heading head."""
def complete_teacher_action(source,base,latent,context):
    return source.layerwise_actor(base,latent)+source.heading_delta(latent,context)
