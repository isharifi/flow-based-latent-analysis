from models.nvp_with_shift import nvp_with_shift


@nvp_with_shift
def make_GLOW(gan_type, graph_dir, framework="pytorch"):
    if framework == "tensorflow":
        from models.GLOW.load import load_from_graph
        generator = load_from_graph(graph_dir)
        return generator
    else:  # framework is PyTorch
        from models.GLOW_pytorch.builder import build
        from models.GLOW_pytorch.config import JsonConfig
        if gan_type == 'GLOW_pt_celeba':
            hparams = JsonConfig("models/GLOW_pytorch/hparams/celeba.json")
        elif gan_type == 'GLOW_pt_anime':
            hparams = JsonConfig("models/GLOW_pytorch/hparams/anime.json")
        generator = build(hparams, False)["graph"]
        generator.dim_shift = generator.flow.output_shapes[-1][1] * generator.flow.output_shapes[-1][2] * \
                              generator.flow.output_shapes[-1][3]
        generator.dim_z = [generator.flow.output_shapes[-1][1], generator.flow.output_shapes[-1][2],
                           generator.flow.output_shapes[-1][3]]
        return generator
