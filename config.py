model_config = {
    "embedding": {
        "num_embeddings": 512,
        "embedding_dim": 1024
    },
    "encoder": {
        "lstm_first": {
            "input_size": 1024,
            "hidden_size": 1024,
            "num_layers": 1,
            "batch_first": True,
            "dropout": 0.1, # no info
            "bidirectional": True
        },
        "lstm_second": {
            "input_size": 2048,
            "hidden_size": 1024,
            "num_layers": 1,
            "batch_first": True,
            "dropout": 0.1, # no info
            "bidirectional": True
        },
        "latent_Wmu": {
            "in_features": 2048,
            "out_features": 512,
            "bias": True
        },
        "latent_Wsig": {
            "in_features": 2048,
            "out_features": 512,
            "bias": True
        },
    },
    "decoder": {
        "W_cond_init_state": {
            "in_features": 512,
            "out_features": 2048,
            "bias": True
        },
        "lstm_conductor": {
            "input_size": 512,
            "hidden_size": 512,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.1, # no info
            "bidirectional": False,
        },
        "W_dec_init_state": {
            "in_features": 512,
            "out_features": 4096,
            "bias": True
        },
        "lstm_decoder": {
            "input_size": 2048,
            "hidden_size": 1024,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.1, # no info
            "bidirectional": False,
        },
        "token_embedding": {
            "num_embeddings": 512,
            "embedding_dim": 1024
        },
        "W_dist_out": {
            "in_features": 1024,
            "out_features": 512,
            "bias": True
        },
        "num_conductor_out": 4,
        "num_decoder_out": 16
    }
}