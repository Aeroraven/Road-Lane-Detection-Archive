<script setup>
import {ref, onBeforeUnmount, nextTick} from 'vue'
import axios from "axios";
import io from 'socket.io-client'
import {Platinum} from '../utils/Platinum'
const playSrc = ref("noimg.png")
const imgSrc = ref("noimg.png")
const fullscreen = ref(false)
const active = ref(false)
const delayedFrames = ref(0)
const sendInterval = ref(150)
const delayedPacket = ref(5)
const serverBusy = ref(false)
const playObject = ref(null)
const activate = () => {
  active.value = true
}
function setFullScreen(){
  fullscreen.value=true
}
function screenShot(id){
  var player = document.getElementById(id);   
  player.setAttribute("crossOrigin", "anonymous"); 
  var canvas = document.createElement("canvas");
  var img = document.createElement("img");
  canvas.width = player.clientWidth;
  canvas.height = player.clientHeight;
  canvas.getContext("2d").drawImage(player, 0, 0, canvas.width, canvas.height);
  var dataURL = canvas.toDataURL("image/png");  
  return dataURL
}
/*
var socket = io.connect(localStorage.getItem("altina_socket_server"));
let lastTime = Date.now()
socket.on('connect', function() {
  socket.emit('connect_start', {"data": 'I\'m connected!'});
  
});
socket.on('server_response',function(message){
  imgSrc.value = "data:image/jpeg;base64,"+message
  
  lastTime=Date.now()
  socket.emit('video_upload',{"data":screenShot('alwen-webcam')})
  console.log("Time Elapsed",Date.now()-lastTime)
  
})

onBeforeUnmount(()=>{
  socket.disconnect()
})*/

async function join(){
  axios({
      url:localStorage.getItem("altina_rapi_server")+"/api/camJoin",
      method:'POST',
      headers:{
        'Content-Type':'undefined'
      },
  }).then(response=>{
    let xid = response.data.data
    navigator.mediaDevices.enumerateDevices().then(function(devices) {
      let cameras = [];
      devices.forEach(function(device) {
        'videoinput' === device.kind && cameras.push(device.deviceId);
      });
      let constraint = true;
      if(cameras.length==2){
        constraint={
          facingMode: "environment"
        }
      }
      navigator.mediaDevices.getUserMedia({
        video: constraint,
      })
      .then((success) => {
        document.getElementById('alwen-webcam').srcObject = success;
        document.getElementById('alwen-webcam-2').srcObject = success;
        document.getElementById('alwen-webcam').play();
        document.getElementById('alwen-webcam-2').play();
        playObject.value = success
        //window.alert(response.data.data)
        imgSrc.value = localStorage.getItem("altina_rapi_server")+"/api/camShow?id="+xid,
        posting(xid)
      })
      .catch((error) => {
        window.alert("摄像头开启失败，请检查摄像头是否可用！"+error)
        console.error("摄像头开启失败，请检查摄像头是否可用！");
      });
    });
    
  }).catch(err=>{
    alert("连接到服务器时出现错误:"+err)
  })
}

async function posting(xid){
  let i = 0
  while(true){
    i+=1
    let res = await axios({
      url:localStorage.getItem("altina_rapi_server")+"/api/camRemain?id="+xid,
      method:"POST",
      headers:{
        'Content-Type':'undefined'
      },
    })
    delayedFrames.value = res.data.data
    if(res.data.data<=delayedPacket.value){
      serverBusy.value = false
      let ts = setInterval(()=>{
        //console.log("SENDING",i)
        axios({
          url:localStorage.getItem("altina_rapi_server")+"/api/camUpload?id="+xid+"&time="+Date.now(),
          method:"POST",
          headers:{
            'Content-Type':'undefined'
          },
          data:Platinum.jsonToFormData({"image":screenShot('alwen-webcam')})
        }).then(()=>{

        })
      },sendInterval.value)
      await new Promise((resolve)=>{
        setTimeout(()=>{
          resolve()
        },400)
      })
      clearInterval(ts)
    }else{
      serverBusy.value = true
      await new Promise((resolve)=>{
        setTimeout(()=>{
          resolve()
        },400)
      })
    }
  }
}
join()

</script>

<template>
  <div class="about">
    <div class="alwen-mobile-only">
      <div style="position:fixed;right:10%;top:70%;z-index:999" @click="setFullScreen">
        全屏展示 >
      </div>
      <div v-show="fullscreen" style="position:fixed;top:0px;height:100%;width:100%;z-index:9999;background-color:black;text-align:center">
        <div style="font-weight:normal;padding-top:10px;position:fixed;top:0px;padding-left:5px;z-index:99;background-color:#333333aa;padding-right:5px;">延迟帧数: {{delayedFrames}} | 延迟时间: {{delayedFrames*sendInterval}} ms | {{serverBusy?`服务器繁忙`:``}}</div>
        <video id="alwen-webcam-2" style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);height:100% !important;aspect-ratio:16/9 !important;" />
        <img :src="imgSrc" id="alwen-webcam-img-b" style="aspect-ratio:16/9;height:100%;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%)">
        <div style="position:fixed;right:1%;bottom:1%"  @click="fullscreen=false">
          退出全屏 >
        </div>
      </div>
    </div>
    <div class="alwen-triangle-page-title">
      <div style="position:relative;z-index:99">
        <span class="alwen-triangle-right-pr" style="z-index:99;display:inline-block">
          <div style="padding-top:10px;padding-left:20px;padding-right:20px;z-index:99;">
            
            <b>实时识别</b> <small>Realtime Recognition</small>
          </div>
        </span>
        <span class="alwen-triangle-right">

        </span>
        <span style="width:100%;background-color:#efefef;height:4px;display:inline-block;transform:translateX(-12px);z-index:0;position:absolute;top:40px;">
          
        </span>
      </div>
    </div>
    <div class="alwen-menu-select alwen-menu-select-ar" style="width:100%;">
      <center style="text-align:center">
        <div class="alwen-bordered" style="display:inline-block;width:50%;" >
          <div style="position:relative">
            <video id="alwen-webcam"  width="800" style="positon:absolute;width:100%;aspect-ratio:16/9 !important;" />
            <img id="alwen-webcam-img" v-if="!fullscreen" :src="imgSrc" style="aspect-ratio:16/9;width:100%;position:absolute;transform:translate(-100%,0%)"/>
          </div>
        </div>
        <div style="font-weight:normal;padding-top:10px">延迟帧数: {{delayedFrames}} | 延迟时间: {{parseInt(delayedFrames*sendInterval*0.25)}} ms</div>
      </center>
      <div class="alwen-menu-tip" style="margin-top:3%">
        <div href="#yolo" class="alwen-skew" style="margin-bottom:3px;">
          <div>实时识别功能需要访问相机才能运行，请检查权限是否授予</div>
        </div>
         <br/> 如果出现无法显示，播放卡顿，请查阅 <u @click="activate" style="cursor:pointer">其他运行时相关问题</u> 进行解决
        
      </div>
    </div>
    <n-drawer v-model:show="active" :width="502" style="background-color:#333333;" placement="bottom">
      <n-drawer-content>
        <template #header>
          <span style="color:white">
            实时识别相关问题
          </span> 
        </template>
        <div style="color:white">
          <b>浏览器支持</b><br/>
          至少确保使用以下版本或更新的浏览器进行访问 <span style="color:#aaaaaa">(标#的标识该浏览器WebCAM无法在安全环境中运行)</span><br/>
          Chrome(74), &nbsp;Edge(79),  &nbsp;FireFox(69), &nbsp; Android WebView(47),  &nbsp;Safari(5), &nbsp; Opera(34,#), &nbsp; Firefox Android(36,#)<br/>
          IE浏览器不被支持。不建议使用除以上浏览器外的浏览器（或上述浏览器的非相同开发商开发的派生浏览器）进行访问，上述浏览器外的浏览器可能出现意外的不正常情况。
        </div><br/>
        <div style="color:white">
          <b>卡顿问题</b><br/>
          如果出现长时间卡顿，可能是由于连接中断造成的，此时可尝试刷新页面。
          由于使用Socket和服务器进行实时传输，速度受到CPU\GPU\TPU性能、Socket配置和网络速度的影响。若需要获得更高速度，可更换网络连接，关闭其他无用的进程。如果您正在使用服务端尝试将服务端的推理模式更改为CUDA或TensorRT，或更改SocketIO的相关配置设置，或寻求服务
          或云服务器提供商提高带宽设置，或更换更高性能的GPU设备。
        </div><br/>
        <div style="color:white">
          <b>无法显示</b><br/>
          如果使用PC，请查阅“设备管理器”确保您的设备的摄像头已经启用。如果使用移动端，请检查对浏览器是否授权访问相机。
        </div><br/>
        
      </n-drawer-content>
    </n-drawer>
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
video{
  width: 100%;
  aspect-ratio: 16/9;
}
</style>