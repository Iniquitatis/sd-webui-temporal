import {getRequest} from "../scripts/utils/requests.js";

export let blendModes = {};

export let models = [];

export let optionCategories = {};

export let pipelineModules = {};

export let samplers = [];

export let schedulers = [];

export let vaes = [];

export let videoFilters = {};

export async function initializeData() {
    blendModes = await getRequest("/temporal/schema/blend_modes");
    models = await getRequest("/temporal/schema/models");
    optionCategories = await getRequest("/temporal/schema/option_categories");
    pipelineModules = await getRequest("/temporal/schema/pipeline_modules");
    samplers = await getRequest("/temporal/schema/samplers");
    schedulers = await getRequest("/temporal/schema/schedulers");
    vaes = await getRequest("/temporal/schema/vaes");
    videoFilters = await getRequest("/temporal/schema/video_filters");
};
