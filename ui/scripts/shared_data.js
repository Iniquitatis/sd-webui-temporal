import {getRequest} from "../scripts/utils/requests.js";

export class SharedData {
    constructor() {
        this.projectName = "";
    }
}

export let shared = new SharedData();

export let objectTypes = {};

export let blendModes = {};

export let pipelineModules = {};

export let videoFilters = {};

export async function initializeData() {
    for (let name of await getRequest("/temporal/object/types")) {
        objectTypes[name] = await getRequest(`/temporal/object/${name}/schema`);
    }

    for (let name of await getRequest("/temporal/object/temporal.blend_modes.BlendMode/subtypes")) {
        blendModes[name] = (await getRequest(`/temporal/object/${name}/schema`)).name;
    }

    for (let name of await getRequest("/temporal/object/temporal.pipeline_module.PipelineModule/subtypes")) {
        pipelineModules[name] = await getRequest(`/temporal/object/${name}/schema`);
    }

    for (let name of await getRequest("/temporal/object/temporal.video_filter.VideoFilter/subtypes")) {
        videoFilters[name] = await getRequest(`/temporal/object/${name}/schema`);
    }
}
