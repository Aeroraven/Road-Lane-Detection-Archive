<script setup>
import router from '../router'
import {proton_main} from '../utils/ProtonDraw.js'
import {onMounted, nextTick, ref} from 'vue'
const showModal = ref(false)
const showModal2 = ref(false)
function goImage(){
  router.push({path:"/image"})
}
function goVideo(){
  router.push({path:"/video"})
}
function goAR(){
  router.push({path:"/ar"})
}

onMounted(() => {
  nextTick(()=>{
    proton_main()
    if(window.location.protocol=="https:"){
      showModal2.value=true
    }
  })
})

</script>

<template>
  <main>
    <div class="alwen-phone-only">
      <div class="alwen-triangle-page-title">
        <span class="alwen-triangle-right-pr">
          <div style="padding-top:10px;padding-left:20px;padding-right:20px">
            <b>平台主页</b> <small>homepage</small>
          </div>
        </span>
        <span class="alwen-triangle-right">

        </span>
        <span style="width:100%;background-color:#efefef;height:4px;display:inline-block;transform:translateX(-12px);z-index:0;position:absolute;top:109px;">
          
        </span>
      </div>
      <div class="alwen-menu-select">
        <n-grid x-gap="15" :cols="5">
          <n-gi>

          </n-gi>
          <n-gi>
            <span class="alwen-number-counter">
              #01
            </span><br/>
            <div class="alwen-menu-alternative" @click="goImage">
              <n-icon size="96">
                <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 512 512" enable-background="new 0 0 512 512" xml:space="preserve"><g><path d="M457.6,140.2l-82.5-4l-4.8-53.8c-1-11.3-11.1-19.2-22.9-18.3L51.5,88.4c-11.8,1-20.3,10.5-19.4,21.7l21.2,235.8
        c1,11.3,11.2,19.2,22.9,18.3l15-1.2l-2.4,45.8c-0.6,12.6,9.2,22.8,22.4,23.5L441.3,448c13.2,0.6,24.1-8.6,24.8-21.2L480,163.5
        C480.6,151,470.7,140.8,457.6,140.2z M102.7,145.5l-7.1,134.8L78.1,305L62,127c0-0.2,0-0.3,0-0.5s0-0.3,0-0.5c1-5,4.4-9,9.6-9.4
        l261-21.4c5.2-0.4,9.7,3,10.5,7.9c0,0.2,0.3,0.2,0.3,0.4c0,0.1,0.3,0.2,0.3,0.4l2.7,30.8l-219-10.5
        C114.2,123.8,103.3,133,102.7,145.5z M436.7,382.4l-84.8-99.5l-37.4,34.3l-69.2-80.8L122.6,367.1L133,168c0,0,0-0.3,0-0.4
        c1-5.4,6.2-9.3,11.9-9l291.2,14c5.8,0.3,10.3,4.7,10.4,10.2c0,0.2,0.3,0.3,0.3,0.5L436.7,382.4z"></path><path d="M384,256c17.6,0,32-14.4,32-32s-14.3-32-32-32c-17.6,0-32,14.3-32,32S366.3,256,384,256z"></path></g></svg>
                </n-icon><br/>
              <span class="alwen-menu-title">
                图片识别
              </span><br/>
              <span class="alwen-menu-title-s">
                Image Recognition
              </span>
            </div>
          </n-gi>
          <n-gi>
            <span class="alwen-number-counter">
              #02
            </span><br/>
            <div class="alwen-menu-alternative"  @click="goVideo">
              <n-icon size="96">
                <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><g fill="none"><path d="M2 6.25A3.25 3.25 0 0 1 5.25 3h13.5A3.25 3.25 0 0 1 22 6.25v11.5A3.25 3.25 0 0 1 18.75 21H5.25A3.25 3.25 0 0 1 2 17.75V6.25zm7.5 3.134v5.231c0 .57.61.932 1.11.658l4.786-2.616a.75.75 0 0 0 0-1.316L10.61 8.726a.75.75 0 0 0-1.11.658z" fill="currentColor"></path></g></svg>
              </n-icon><br/>
              <span class="alwen-menu-title">
                视频识别
              </span><br/>
              <span class="alwen-menu-title-s">
                Video Recognition
              </span>
            </div>
          </n-gi>
          <n-gi>
            <span class="alwen-number-counter">
              #03
            </span><br/>
            <div class="alwen-menu-alternative"  @click="goAR">
              
              <n-icon size="96">
                <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 32 32"><path d="M11.61 29.92a1 1 0 0 1-.6-1.07L12.83 17H8a1 1 0 0 1-1-1.23l3-13A1 1 0 0 1 11 2h10a1 1 0 0 1 .78.37a1 1 0 0 1 .2.85L20.25 11H25a1 1 0 0 1 .9.56a1 1 0 0 1-.11 1l-13 17A1 1 0 0 1 12 30a1.09 1.09 0 0 1-.39-.08zM17.75 13l2-9H11.8L9.26 15h5.91l-1.59 10.28L23 13z" fill="currentColor"></path></svg>
              </n-icon><br/>
              <span class="alwen-menu-title">
                实时识别
              </span><br/>
              <span class="alwen-menu-title-s">
                Realtime Recognition
              </span>
            </div>
          </n-gi>
          <n-gi>
            
          </n-gi>
        </n-grid>
        <div class="alwen-menu-tip">
          <div href="#yolo" class="alwen-skew">
            <div>请选择其中一种识别方式。</div>
          </div>
        </div>
        
      </div>
    </div>
    <div class="alwen-pc-only">
      <canvas id="testCanvas" class="alwen-proton-wgl"></canvas>
      <div class="alwen-proton-title">
        <span class="alwen-proton-title-v animate__animated animate__fadeIn cust_delay_2">
          车道线识别演示平台
        </span> 
        <br/>
        <span class="alwen_ak_non_bold animate__animated animate__fadeIn cust_delay_2">Lane Detection Visualization Platform</span>
        <div class="alwen_ak_hr">

        </div>
        <div class="alwen-proton-right-sb animate__animated animate__fadeIn cust_delay_2">
          此页面为同济大学软件工程专业方向综合设计(机器智能方向)课程设计项目的演示平台界面。您可以凭借此应用完成车道标线以及导向箭头等多个区域的识别。
          目前应用支持包括图片识别（上传单张图片）、视频识别(上传完整视频后以流式结果展示)、实时识别(通过摄像头等相机设备进行实时识别)在内的三种功能。
        </div>
      </div>
      <div class="alwen-left-menu">
        <div style="text-align:center">
          <div class="alwen-skew-c alwen_ak_loading_prog_logo_sub" style="width:60%;display:inline-block;height:40px;margin-bottom:80px">
            <div style="position:relative;top:2px;font-size:20px;font-family: 'Geometos','SourceHanSerifHeavy',sans-serif;">主菜单 Menu</div>
          </div>
        </div>
        
        <div class="alwen-left-menu-choice" @click="goImage">
          <div class="animate__animated animate__slideInLeft animate__slow cust_delay_0" style="display:inline-block">
            <div class="animate__animated animate__slideInLeft animate__slow cust_delay_0" style="display:inline-block">
                <div class="alwen-left-menu-choice-sr" style="display:inline-block;z-index:99;position:relative">
                  >> &nbsp;
                </div>
              <span class="alwen-left-menu-choice-text" style="z-index:99;position:relative">图像识别 Imgae Recognition</span>
              <a class="alwen-hollow">Image Recognition</a>
            </div>
          </div>
          
        </div>
        <div class="alwen-left-menu-hr">
        </div>
        <div class="alwen-left-menu-choice" @click="goVideo">
          <div class="animate__animated animate__slideInLeft animate__slow cust_delay_1" style="display:inline-block">
            <div class="animate__animated animate__slideInLeft animate__slow cust_delay_1" style="display:inline-block">
                <div class="alwen-left-menu-choice-sr" style="display:inline-block;z-index:99;position:relative">
                  >> &nbsp;
                </div>
              <span class="alwen-left-menu-choice-text" style="z-index:99;position:relative">视频识别 Video Recognition</span>
              <a class="alwen-hollow">VIDEO Recognition</a>
            </div>
          </div>
        </div>
        <div class="alwen-left-menu-hr">
        </div>
        <div class="alwen-left-menu-choice" @click="goAR">
          <div class="animate__animated animate__slideInLeft animate__slow cust_delay_2" style="display:inline-block">
            <div class="alwen-left-menu-choice-sr" style="display:inline-block;z-index:99;position:relative">
              >> &nbsp;
            </div>
            <span class="alwen-left-menu-choice-text" style="z-index:99;position:relative">实时识别 Realtime Detection</span>
            <a class="alwen-hollow">REALITME DETECTION</a>
          </div>
        </div>
        <div class="alwen-left-menu-hr">
        </div>
        <div class="alwen-left-menu-choice" @click="showModal=true">
         
          <div class="animate__animated animate__slideInLeft animate__slow cust_delay_3" style="display:inline-block">
            <div class="animate__animated animate__slideInLeft animate__slow cust_delay_1" style="display:inline-block">
              <div class="alwen-left-menu-choice-sr" style="display:inline-block;z-index:99;position:relative">
                >> &nbsp;
              </div>
              <span class="alwen-left-menu-choice-text" style="z-index:99;position:relative">帮助和支持 Help & Support</span>
              <a class="alwen-hollow">Help & Support</a>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <n-modal v-model:show="showModal" class="alwen-modal" style="background-color: #2f2f2f !important; max-height:80%;">
      <n-card
        style="width: 600px; "
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <template #header>
          <span class="alwen-modal-title">
            <span style="width:10px;display:inline-block;background-color:#efefef">
              &nbsp;
            </span>
            帮助与支持 Help & SUpport
          </span>
        </template>
        <template #header-extra>
          噢！
        </template>
        <div class="scroll_config" style="font-family:sans-serif !important; max-height: 400px !important; overflow-y:scroll;padding-right:4px">
          帮助和支持包括作者信息、浏览器要求和开放源代码许可信息及对引用资料的标注 <br/><br/>
          <b><big># 作者信息</big></b>
          <li> Team Collaborators: Huang Zhiwen(1950641), Lu Han (1950062) and Ma Haoyang(1950994) </li>
          <li> This project serves as the part of Tongji University School Software Enginnering course design. </li>
          <li> 本项目前端使用Vue3+Webpack5+Babel构建, 使用Cordova+AndroidStudio基于AndroidSDK 30和生成移动端发布版本</li>
          <li> 本项目使用Github作为项目托管平台：<a href="https://github.com/Aeroraven/RoadLane" style="color:#ff9900">https://github.com/Aeroraven/RoadLane</a></li>
          <br/><br/>
          <b><big># 浏览器要求</big></b>
          <br/>
          <li><b style="color:#ff9900">基本要求</b>: 请使用Edge, Chrome, Safari, FireFox四者之一的浏览器进行访问。
            <ul>
              <li>不支持的浏览器: Internet Explorer、Opera Mini、UC浏览器、百度浏览器。(该类型浏览器对Vue3支持尚待确认)</li>
              <li>其他浏览器: 360浏览器、搜狗浏览器、华为浏览器等其他浏览器。(使用该类型浏览器需核对浏览器内核版本)</li>
            </ul>
          </li>
          <li><b style="color:#ff9900">浏览网页</b>: 至少要求浏览器支持ECMAScript 2015标准。确保浏览器满足下列版本要求。
            <ul>
              <li>Edge: >=79</li>
              <li>Chrome: >=63</li>
              <li>Safari: >=11.1</li>
              <li>FireFox: >=67</li>
            </ul>
          </li>
          <li><b style="color:#ff9900">推理设备</b>: 至少要求浏览器支持WebGPU, WebASM，WebGL的其中一个。确保浏览器满足下列版本要求。
            <ul>
              <li>Edge: >=15</li>
              <li>Chrome: >=58</li>
              <li>Safari: >=10.1</li>
              <li>FireFox: >=53</li>
            </ul>
          </li>
          <li><b style="color:#ff9900">WebCAM</b>: 至少要求浏览器支持navigator.mediaDevices。确保浏览器满足下列版本要求。
            <ul>
              <li>Edge: >=12 (若要确保隐私安全，则>=79)</li>
              <li>Chrome: >=47 (若要确保隐私安全，则>=74)</li>
              <li>Safari: >=11</li>
              <li>FireFox PC: >=36 (若要确保隐私安全，则>=69)</li>
              <li>FireFox Android: >=36 (不支持安全语境)</li>
            </ul>
          </li>
          
          <br/><br/>
          
          <b><big># 开放源代码许可/代码和依赖</big></b>
          <li><b style="color:#ff9900">OpenCV.js [OpenCV] (Apache2.0)</b>: https://github.com/opencv/opencv/blob/4.x/LICENSE </li>
          <li><b style="color:#ff9900">Vue [Vuejs] (MIT)</b>: https://github.com/vuejs/core/blob/main/LICENSE </li>
          <li><b style="color:#ff9900">Vue-Router [Vuejs] (MIT)</b>: https://github.com/vuejs/router/blob/main/LICENSE </li>
          <li><b style="color:#ff9900">Pinia [Vuejs] (MIT)</b>: https://github.com/vuejs/pinia/blob/v2/LICENSE </li>
          <li><b style="color:#ff9900">Animate.css [Animate-css] (Hippocratic License)</b>: https://github.com/animate-css/animate.css/blob/main/LICENSE </li>
          <li><b style="color:#ff9900">Naive-UI [TuSimple] (MIT)</b>: https://github.com/TuSimple/naive-ui/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">Proton Engine [drawcall] (MIT)</b>: https://github.com/drawcall/Proton/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">Axios [Axios] (MIT)</b>: https://github.com/axios/axios/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">SASS [Sass] (MIT)</b>: https://github.com/sass/sass/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">Socket.IO-Client [Socketio] (MIT)</b>: https://github.com/socketio/socket.io-client/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">Vite [Vitejs] (MIT)</b>: https://github.com/vitejs/vite/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">ONNXRuntime [Microsoft] (MIT)</b>: https://github.com/microsoft/onnxruntime/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">Style-Loader [JS Foundation] (MIT)</b>: https://github.com/webpack-contrib/style-loader/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">Node-SASS [Andrew Nesbitt] (MIT)</b>: https://github.com/sass/node-sass/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">babel-loader [Luís Couto] (MIT)</b>: https://github.com/babel/babel-loader</li>
          <li><b style="color:#ff9900">html-webpack-plugin [JS Foundation] (MIT)</b>: https://github.com/jantimon/html-webpack-plugin/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">TSParticles [Matteo Bruni] (MIT)</b>: https://github.com/matteobruni/tsparticles/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">CSS-Loader [JS Foundation] (MIT)</b>: https://github.com/webpack-contrib/css-loader/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">Vue-Loader [You Yuxi] (MIT)</b>: https://github.com/vuejs/vue-loader/blob/next/LICENSE</li>
          <li><b style="color:#ff9900">Webpack [JS Foundation] (MIT)</b>: https://github.com/webpack/webpack/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">Codrova-Android [Apache] (Apache2.0)</b>: https://github.com/apache/cordova-android/blob/master/LICENSE</li>
          
          <br/><br/>
          <b><big># 开放源代码许可/其他</big></b>
          <li><b style="color:#ff9900">Source Han Sans/思源黑体 [Adobe/Google] (OFL)</b>: https://github.com/adobe-fonts/source-han-sans/blob/master/LICENSE.txt</li>
          <li><b style="color:#ff9900">Source Han Serif/思源宋体 [Adobe/Google] (OFL)</b>: https://github.com/adobe-fonts/source-han-serif/blob/master/LICENSE.txt</li>
          <li><b style="color:#ff9900">Geometos [Deepak Dogra] (非商用)</b>: https://www.dafont.com/geometos.font</li>
          <li><b style="color:#ff9900">Novenceto Sans [Jan Tonellato]</b>: https://fonts.adobe.com/fonts/novecento-sans#licensing-section</li>
          <li><b style="color:#ff9900">IonIcons [Ionic-team]</b>: https://github.com/ionic-team/ionicons/blob/main/LICENSE</li>
          <li><b style="color:#ff9900">Fluent UI System Icons [Microsoft]</b>: https://github.com/microsoft/fluentui-system-icons/blob/master/LICENSE</li>
          <li><b style="color:#ff9900">Carbon [Carbon Design System]</b>: https://github.com/carbon-design-system/carbon/blob/main/LICENSE</li>
        
        </div>
        
      </n-card>
    </n-modal>
    <n-modal v-model:show="showModal2" class="alwen-modal" style="background-color: #2f2f2f !important; max-height:80%;">
      <n-card
        style="width: 600px; "
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <template #header>
          <span class="alwen-modal-title">
            <span style="width:10px;display:inline-block;background-color:#efefef">
              &nbsp;
            </span>
            不支持HTTPS
          </span>
        </template>
        <template #header-extra>
          <a style="color:#efefef" @click="showModal2=false">X</a>
        </template>
        <div class="scroll_config" style="font-family:sans-serif !important; max-height: 400px !important; overflow-y:scroll;padding-right:4px">
          <!--由于安全因素，使用浏览器访问应用时，需要相应的CA签发的证书，如您需要继续，则需要下载CA证书并添加到“受信任的根证书颁发机构”。<a href="https://aeroraven.github.io/altina-backend/auth_ca.crt" style="color:#ff9900">下载CA证书</a><br/>
          您也可以选择以下的解决方案:<br/>
          <li>对于Windows/Linux/MacOS: 使用Electron发布版本或在回环地址上运行应用</li>
          <li>对于Android: 使用Cordova发布版本</li>
          <li>对于iOS: 更换为Windows/Linux/macOS/Android或安装CA证书</li>-->
          目前的应用不支持以HTTPS协议运行。请将使用HTTP协议重新运行服务器。
        </div>
      </n-card>
    </n-modal>
  </main>
</template>
<style>
.alwen-hollow{
  position:absolute;
  right:0px;
  top:0px;
  font-size: 45px;
  font-weight: bolder;
  -webkit-text-stroke:0.2px #222222;
  color: transparent !important;
  transform: translate(20px,-18px);
  z-index:0;
  white-space:nowrap;
  text-align:right;
  transition: all 0.3s;
}
.alwen-left-menu-choice:hover .alwen-hollow{
  -webkit-text-stroke:0.2px #3a3a3a;
  color:transparent !important;
  transition: all 0.3s;
}
::-webkit-scrollbar {
       width:5px;
       height:5px
    }
::-webkit-scrollbar-thumb {
        background-color:#ccc;
        border:solid 1px #ccc;
        border-radius:2px;
    }
::-webkit-scrollbar-track {
        --webkit-box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
        border-radius: 2px;
        background: #EDEDED;
    }
::-webkit-scrollbar-arrow {
        color:#F00;
        background:#0F0;
}    
.n-card-header__main{
  color:#efefef !important;
}
.alwen-left-menu-choice > .alwen-left-menu-choice-sr{
  color: #ff000000;
}
.alwen-left-menu-choice:hover > .alwen-hollow{
  color: transparent ;
  font-size:999px !important;
}
.cust_delay_0 {
  --animate-duration: 0.4s;
}
.cust_delay_1 {
  --animate-duration: 0.8s;
}
.cust_delay_2 {
  --animate-duration: 1.2s;
}
.cust_delay_3 {
  --animate-duration: 1.6s;
}
.alwen-triangle-page-title-phone{
  position:fixed;
  bottom:0px;
  left:5px;
}
</style>