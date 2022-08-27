import { cvWrapper } from "./OpenCV";
import * as AlBackend from './AltinaBackend'
import * as ort from 'onnxruntime-web'
import * as WebDNN from 'webdnn'
var cv = cvWrapper.cv;

export class Altina{
    constructor(){
        this.modelSessions = {};
        this.backend = null;
        
    }
    async init(){
        await AlBackend.default() 
        this.backend = AlBackend
        return this
    }
    static async getInst(){
        let inst = null;
        if(!inst){
            inst = await (new Altina()).init()
        }
        return inst
    }
    /* Asynchronized version of cv.imread
       @param imageElementId. Must be the element id of a DOM.
    */
    altinaImRead(imageElementId){
        return new Promise((resolve,reject)=>{
            let element = document.getElementById(imageElementId)
            console.log(element)
            if(element==null){
                reject("Element does not exist")
            }else{
                if(element.complete){
                    resolve(cv.imread(element))
                }else{
                    element.onload = ()=>{
                        resolve(cv.imread(element))
                    }
                }
            }
        })
    }
    /* Read an image from cv::Mat and perform ImageNet normalization and reshaping
      @param imMatrix: Must be cv::Mat with 3 Channels (OpenCV)
    */
    altinaImPreproc(imMatrix){
        let x = new Float32Array(imMatrix.cols * imMatrix.rows * 3)
        return this.backend.altina_image_to_tensor_preproc(imMatrix.data,x,imMatrix.rows,imMatrix.cols)
    }
    /* Read an image from cv::Mat and perform ImageNet normalization and reshaping
      @param imMatrix: Must be cv::Mat with 4 Channels (OpenCV)
    */
    altinaImPreprocResized(imMatrix,imHeight,imWidth){
        let imMatrixAltered = new cv.Mat()
        cv.resize(imMatrix,imMatrixAltered,new cv.Size(imWidth,imHeight),0,0,cv.INTER_AREA)
        let x = new Float32Array(imMatrixAltered.cols * imMatrixAltered.rows * 3)
        let ret = this.backend.altina_image_to_tensor_preproc(imMatrixAltered. data,x,imMatrixAltered.rows,imMatrixAltered.cols)
        return ret
    }
    altinaImToTensor(imNormalizedMatrix, imHeight, imWidth){
        return new ort.Tensor("float32", imNormalizedMatrix, [1,3,imHeight,imWidth]);
    }
    async altinaCreateInferenceSession(modelName, modelPath,inferenceDevice=['wasm']){
        if(modelName in this.modelSessions){
            return this.modelSessions[modelName]
        }else{
            this.modelSessions[modelName] = await ort.InferenceSession.create(modelPath,{ executionProviders: inferenceDevice, graphOptimizationLevel: 'all' })
        }
        return this.modelSessions[modelName]
    }
    async altinaInfer(modelName, inputData){
        let feeds = {}
        feeds[this.modelSessions[modelName].inputNames[0]] = inputData
        console.log(feeds)
        return await this.modelSessions[modelName].run(feeds)
    }
    async altinaIntegratedImgProc(imageElementId,imgHeight,imgWdith){
        let image = await this.altinaImRead(imageElementId)
        image = this.altinaImPreprocResized(image,imgHeight,imgWdith)
        image = this.altinaImToTensor(image,imgHeight,imgWdith)
        return image
    }
    debug(){
        
    }
    async debug2(){
        console.log("Creating inference session")
        let session = await this.altinaCreateInferenceSession("debug","/models/arrowfcn.onnx")
        console.log("Loading image")
        let image = await this.altinaIntegratedImgProc("test",480,800)
        console.log("Infering")
        console.log(Date.now())
        let result;
        for(let i = 0;i<5;i++){
            result = await this.altinaInfer("debug",image)
        }
        console.log(Date.now())
        console.log("Completed")
        console.log(result)
    }
    async debug3(){
        console.log("Creating inference session")
        let session = await this.altinaCreateInferenceSession("debug","/models/yolov5s.onnx")
        console.log("Loading image")
        let image = await this.altinaIntegratedImgProc("test",640,640)
        console.log("Infering")
        console.log(Date.now())
        let result;
        for(let i = 0;i<5;i++){
            result = await this.altinaInfer("debug",image)
        }
        console.log(Date.now())
        console.log("Completed")
        console.log(result)
    }
    async debugWebDNN(){
        let runner, image, probabilities;
        runner = await WebDNN.load('/models/yolov5s');
        image = runner.inputs[0]; 
        probabilities = runner.outputs[0];
        image.set(await WebDNN.Image.getImageArray('/origin.jpeg'));
        await runner.run(); 
    }
}