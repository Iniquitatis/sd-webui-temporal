import {getRequest} from "../scripts/utils/requests.js";

export let blendModes = {};

export let models = [];

export let optionCategories = {};

export let pipelineModules = {};

export let presets = [];

export let projects = [];

export let samplers = [];

export let schedulers = [];

export let vaes = [];

export let videoFilters = {};

export async function initializeData() {
    blendModes = await getRequest("/temporal/blend_modes");
    models = await getRequest("/temporal/models");
    optionCategories = await getRequest("/temporal/option_categories");
    pipelineModules = await getRequest("/temporal/pipeline_modules");
    presets = await getRequest("/temporal/presets");
    projects = await getRequest("/temporal/projects");
    samplers = await getRequest("/temporal/samplers");
    schedulers = await getRequest("/temporal/schedulers");
    vaes = await getRequest("/temporal/vaes");
    videoFilters = await getRequest("/temporal/video_filters");
};
