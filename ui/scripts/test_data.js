export let blendModes = {};

export let models = {};

export let pipelineModules = {
    "new_processing": {
        icon: "\u{0001f9ec}",
        name: "New Processing",
        parameters: {
            model: {
                name: "Model",
                type: "enum",
                ui_type: "list",
                choices: {
                    "blah_1.safetensors": "blah_1.safetensors",
                    "blah_2.safetensors": "blah_2.safetensors",
                    "blah_3.safetensors": "blah_3.safetensors",
                },
                default: "blah_2.safetensors",
            },
            vae: {
                name: "VAE",
                type: "enum",
                ui_type: "list",
                choices: {
                    "vae_blah_1.safetensors": "vae_blah_1.safetensors",
                    "vae_blah_2.safetensors": "vae_blah_2.safetensors",
                    "vae_blah_3.safetensors": "vae_blah_3.safetensors",
                },
                default: "vae_blah_3.safetensors",
            },
            clip_skip: {
                name: "CLIP skip",
                type: "int",
                ui_type: "slider",
                minimum: 0,
                maximum: 8,
                step: 1,
                default: 0,
            },
            positive_prompt: {
                name: "Positive prompt",
                type: "string",
                ui_type: "area",
                default: "female, portrait, forest, cinematic, backlighting",
            },
            negative_prompt: {
                name: "Negative prompt",
                type: "string",
                ui_type: "area",
                default: "male, gray theme, 2d",
            },
            sampler: {
                name: "Sampler",
                type: "enum",
                ui_type: "list",
                choices: {
                    "dpmpp_2m": "DPM++ 2M",
                    "dpmpp_sde": "DPM++ SDE",
                    "euler": "Euler",
                },
                default: "euler",
            },
            scheduler: {
                name: "Scheduler",
                type: "enum",
                ui_type: "list",
                choices: {
                    "auto": "Automatic",
                    "uniform": "Uniform",
                    "karras": "Karras",
                },
                default: "auto",
            },
            steps: {
                name: "Steps",
                type: "int",
                ui_type: "slider",
                minimum: 1,
                maximum: 150,
                step: 1,
                default: 20,
            },
            cfg: {
                name: "CFG",
                type: "float",
                ui_type: "slider",
                minimum: 1.0,
                maximum: 30.0,
                step: 0.5,
                default: 5.0,
            },
            strength: {
                name: "Denoising strength",
                type: "float",
                ui_type: "slider",
                minimum: 0.0,
                maximum: 1.0,
                step: 0.01,
                default: 0.5,
            },
            seed: {
                name: "Seed",
                type: "seed",
            },
        },
    },
};

export let presets = {};

export let projects = {};

export let samplers = {};

export let schedulers = {};

export let vaes = {};
