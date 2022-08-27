<script setup>
import {ref} from 'vue'
import axios from "axios";
const baseServer = ref(localStorage.getItem("altina_rapi_server"))
const playSrc = ref("noimg.png")
const customRequest = ref(({
  file,
  data,
  headers,
  withCredentials,
  action,
  onFinish,
  onError,
  onProgress
}) => {
  const formData = new FormData();
  if (data) {
  Object.keys(data).forEach((key) => {
    formData.append(key, data[key]);
  });
  }
  formData.append("image", file.file);
  let newReq = axios(
    {
      url:action,
      method:'POST',
      headers:{
        'Content-Type':'undefined'
      },
      data:formData,
      onUploadProgress: (progress) => {
        console.log(Math.ceil(progress))
        onProgress({ percent: Math.ceil(progress.loaded/progress.total*100) });
      }
    }
  ).then((response) => {
    console.log("Success")
    playSrc.value=localStorage.getItem("altina_rapi_server")+"/api/videoInfer?path="+response.data.data
    window.alert("上传成功")
    fileList.value = []
    onFinish();
  }).catch((error) => {
    window.alert("上传错误，请检查网络连接。")
    console.log("Fail")
    fileList.value = []
    onError();
  });

});
const fileList = ref([])
async function beforeUpload (data) {
  if (fileList.value.length>=1) {
    return false
  }
  return true
}

function handleUploadChange(data) {
  console.log(fileList.value)
  window.R=fileList.value
  if(data.fileList.length>1){
    //console.log("Upload Rejected A", data.fileList.length)
    return 
  }
  
  if(fileList.value.length>=1){
    //console.log("Upload Rejected B", fileList.value.length)
    return 
  }
  fileList.value = data.fileList;
}
</script>

<template>
  <div class="about">
    <div style="position:relative;z-index:99">
      <div class="alwen-triangle-page-title">
        <span class="alwen-triangle-right-pr" style="z-index:99;display:inline-block">
          <div style="padding-top:10px;padding-left:20px;padding-right:20px;z-index:99;">
            
            <b>视频识别</b> <small>Video Recognition</small>
          </div>
        </span>
        <span class="alwen-triangle-right">

        </span>
        <span style="width:100%;background-color:#efefef;height:4px;display:inline-block;transform:translateX(-12px);z-index:0;position:absolute;top:40px;">
          
        </span>
      </div>
    </div>
    <div class="alwen-menu-select" style="width:80%;">
      <n-grid x-gap="15" :cols="7" >
        <n-gi span="3">
          <span class="alwen-bordered-title" style="z-index:9">
            视频上传
          </span>
          <div class="alwen-bordered">
            <n-upload
              v-model:file-list="fileList"
              :action="baseServer+`/api/videoUpload`"
              :headers="{
              'naive-info': 'hello!'
              }"
              :data="{
              'naive-data': 'cool! naive!'
              }"
              :custom-request="customRequest"
              style="color:#efefef;position:relative;height:100%"
              @change="handleUploadChange"
              @beforeUpload="beforeUpload"
            >
                <n-upload-dragger style="width:100%;background-color:#333333;height:100%;aspect-ratio:16/7">
                  <div class="alwen-upload-pc">
                    <div style="margin-bottom: 12px">
                      <n-icon size="48" :depth="3" style="color:#ffffff !important" >
                        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="color:#ffffff !important" viewBox="0 0 512 512"><path d="M320 367.79h76c55 0 100-29.21 100-83.6s-53-81.47-96-83.6c-8.89-85.06-71-136.8-144-136.8c-69 0-113.44 45.79-128 91.2c-60 5.7-112 43.88-112 106.4s54 106.4 120 106.4h56" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"></path><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32" d="M320 255.79l-64-64l-64 64"></path><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32" d="M256 448.21V207.79"></path></svg>
                      </n-icon>
                    </div>
                    <span style="font-size: 16px">
                      点击或者拖动文件到该区域来上传
                    </span>
                    <n-p depth="3" style="margin: 8px 0 0 0">
                      请不要上传敏感数据，比如你的银行卡号和密码，信用卡号有效期和安全码
                    </n-p>
                  </div>
                  <div class="alwen-upload-mobile">
                    <div style="margin-bottom: 12px">
                      <n-icon size="36" :depth="3" style="color:#ffffff !important" >
                        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="color:#ffffff !important" viewBox="0 0 512 512"><path d="M320 367.79h76c55 0 100-29.21 100-83.6s-53-81.47-96-83.6c-8.89-85.06-71-136.8-144-136.8c-69 0-113.44 45.79-128 91.2c-60 5.7-112 43.88-112 106.4s54 106.4 120 106.4h56" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32"></path><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32" d="M320 255.79l-64-64l-64 64"></path><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="32" d="M256 448.21V207.79"></path></svg>
                      </n-icon>
                    </div>
                    <span style="font-size: 16px">
                      点击此处选择要上传的文件
                    </span>
                    
                  </div>
                </n-upload-dragger>
            </n-upload>
          </div>
        </n-gi>
        <n-gi span="1" style="text-align:center">
          <div class="alwen-menu-select-v" style="text-align:center; display:inline-block">
            <n-icon size="60" style="transform:translate(-30px,-50px)">
              <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7l5 5l-5 5"></path><path d="M13 7l5 5l-5 5"></path></g></svg>
            </n-icon>
          </div>
        </n-gi>
        <n-gi span="3">
          <span class="alwen-bordered-title" style="z-index:9">
            识别结果
          </span>
          <div class="alwen-bordered">
            <img :src="playSrc" style="width:100%;height:100%;position:relative;top:2px"/>
          </div>
        </n-gi>
      </n-grid>
      <div class="alwen-menu-tip">
        <div href="#yolo" class="alwen-skew">
          <div>受限于网络和设备，上传文件至显示结果有一定延迟。</div>
        </div>
        <br/><br/>
        
      </div>
    </div>
  </div>
</template>

<style>
.alwen-triangle-right{
	display:inline-block;
	width:0;
	height:0;
	border-top: 25px solid transparent;
	border-left: 40px solid #efefef;
	border-bottom: 25px solid transparent;
  transform: translateY(17px);
}
.alwen-triangle-right-pr{
  background-color:#efefef;
  display:inline-block;
  height:50px;
  color: #333333;
  font-size:20px
}
.alwen-bordered{
  aspect-ratio: 16/9;
  width:100%;
  border: #efefef 4px dashed;
  border-radius: 10px;
  text-align:center
}
.alwen-bordered-title{
  position: fixed;
  background-color: #efefef;
  color:#333333;
  padding-left: 20px;
  padding-right:20px;
  padding-top:5px;
  padding-bottom:5px;
  transform: translate3d(-10px,-10px,0px);
}
</style>

<style lang="scss">
//wcnm !important
.n-upload-trigger{
  width:100% !important;
}
.n-upload-file-info__name{
  color: #efefef !important;
  font-size:10px !important;
}
.n-upload-file{
  background-color: #333333;
  transition: all 0.5s !important;
}
.n-upload-file:hover{
  background-color: #777777 !important;
  color: #000000 !important;
  transition: all 0.5s !important;
}
.n-upload-file-info{
  padding-top:0px !important;
  padding-bottom:0px !important;
}
.n-upload-file-list{
  font-size:10px !important;
  margin-top:0px !important;
  padding-bottom:0px !important;
}
.n-progress{
  margin-bottom:0px !important;
  padding-bottom:0px !important;
  
}
</style>