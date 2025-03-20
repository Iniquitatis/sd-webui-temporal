import {getRequest} from "../scripts/utils/requests.js";

export let blendModes = {};

export let optionCategories = {};

export let pipelineModules = {};

export let videoFilters = {};

export async function initializeData() {
    blendModes = await getRequest("/temporal/schema/blend_modes");
    optionCategories = await getRequest("/temporal/schema/option_categories");
    pipelineModules = await getRequest("/temporal/schema/pipeline_modules");
    videoFilters = await getRequest("/temporal/schema/video_filters");
};
