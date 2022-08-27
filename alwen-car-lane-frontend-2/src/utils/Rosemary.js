import axios from "axios"
import { Socket } from "engine.io-client"
import { Exception } from "sass"
import io from "socket.io-client"
export default class Rosemary{
    serverLists = {}
    getChromeVer () {
        var ua = navigator.userAgent;
        var b = ua.indexOf("Chrome/");
        if (b < 0) {
            return 0;
        }
        return parseFloat (ua.substring (b + 8, ua.lastIndexOf("\.")));
    }
    getBroswer(){
        var sys = {};
        var ua = navigator.userAgent.toLowerCase();
        var s;
        (s = ua.match(/edge\/([\d.]+)/)) ? sys.edge = s[1] :
        (s = ua.match(/rv:([\d.]+)\) like gecko/)) ? sys.ie = s[1] :
        (s = ua.match(/msie ([\d.]+)/)) ? sys.ie = s[1] :
        (s = ua.match(/firefox\/([\d.]+)/)) ? sys.firefox = s[1] :
        (s = ua.match(/chrome\/([\d.]+)/)) ? sys.chrome = s[1] :
        (s = ua.match(/opera.([\d.]+)/)) ? sys.opera = s[1] :
        (s = ua.match(/version\/([\d.]+).*safari/)) ? sys.safari = s[1] : 0;
    
        if (sys.edge) return { broswer : "Edge", version : sys.edge };
        if (sys.ie) return { broswer : "IE", version : sys.ie };
        if (sys.firefox) return { broswer : "Firefox", version : sys.firefox };
        if (sys.chrome) return { broswer : "Chrome", version : sys.chrome };
        if (sys.opera) return { broswer : "Opera", version : sys.opera };
        if (sys.safari) return { broswer : "Safari", version : sys.safari };
        
        return { broswer : "", version : "0" };
    }
    constructor(){
    
    }    
    async rosLoader(successHandler,progressHandler,errorHandler = (x)=>{window.alert("发生错误",x)}){
        let ua = navigator.userAgent.toLowerCase(); 
        progressHandler("运行环境检查")
        let info = {
            ie : /msie/ .test(ua) && !/opera/ .test(ua),
            op : /opera/ .test(ua), 
            sa : /version.*safari/.test(ua), 
            ch : /chrome/.test(ua),  
            ff : /gecko/.test(ua) && !/webkit/.test(ua)
        };
        let infos = this.getBroswer()
        if(info.sa==false&&info.ch==false&&info.ff==false){
            progressHandler("当前浏览器或系统配置不受支持,请更换为Chrome,FireFox或Safari内核的浏览器")
            alert("浏览器或系统配置不受支持")
            return
        }
        console.log("A")
        if(info.ch==true||info.ff==true){
            if(parseInt(infos.version)<80){
                progressHandler("浏览器内核或WebView版本("+parseInt(infos.version)+")过低，请升级至Chrome/Firefox或WebView>=80以上版本")
                alert("您使用的浏览器内核版本无法支持应用运行，应用最低支持的内核版本为80。"+
                "对于移动端用户，请在调整网络配置后，在Google Play应用商店中升级系统WebView版本"+
                "(https://play.google.com/store/apps/details?id=com.google.android.webview)。")
                return
            }
        }
            
        if(info.sa==true){
            if(parseInt(infos.version)<15){
                progressHandler("浏览器内核版本过低，请升级至Safari>=15以上版本")
                alert("您使用的浏览器内核版本无法支持应用运行，应用最低支持的内核版本为15。"+
                "对于iOS用户，如果您需要运行本应用，您需要在“设置-General-SoftwareUpdate”升级iOS版本至>=15")
                return
            }
        }

        successHandler()
        let steps = [
            ["加载字体", this.rosFontLoading],
            ["查询可用服务器列表", this.rosServerListing],
            ["测试服务器状态", this.rosServerChecking],
            //["测试Socket状态", this.rosServerCheckingSocket],
            ["初始化完成", this.rosDone]
        ]
        try{
            for(let i = 0;i<steps.length;i++){
            
                progressHandler(steps[i][0])
                await steps[i][1](successHandler,progressHandler,errorHandler)
                successHandler()
            
            }
        }catch(exception){
            errorHandler(exception)
        }
    }
    async rosDone(successHandler,progressHandler,errorHandler){
        
    }
    async rosServerCheckingSocket(successHandler,progressHandler,errorHandler){
        let serverChoices = Rosemary.serverLists.availableSocketBackends
        let bestServer = ""
        let bestServerTimeout = 12000

        for(let i=0;i<serverChoices.length;i++){
            if(i<5){
                successHandler()
            }
            try{
                progressHandler("正在测试Socket("+i+"/"+serverChoices.length+") : "+serverChoices[i])
                let startTimestamp = Date.now()
                let ioInst = io.connect(serverChoices[i])
                let validStatus = false
                ioInst.on("connect",()=>{
                    validStatus = true;
                })
                await new Promise((resolve)=>{
                    setTimeout(()=>{
                        resolve()
                    },3000)
                })
                ioInst.disconnect()
                if(validStatus){
                    bestServer = serverChoices[i]
                    bestServerTimeout = Date.now() - startTimestamp
                }
                
            }catch(err){

            }
            
        }
        if(serverChoices.length<5){
            for(let i=0;i<5-serverChoices.length;i++){
                successHandler()
            }
        }
        if(bestServer===""){
            progressHandler("目前无可用的服务器，至"+serverChoices.length+"个服务器的连接均超时")
            throw new Error("目前无可用的服务器")
        }else{
            localStorage.setItem("altina_socket_server",bestServer)
        }

    }
    async rosServerChecking(successHandler,progressHandler,errorHandler){
        console.log(Rosemary.serverLists)
        let serverChoices = Rosemary.serverLists.availableBackends
        let bestServer = ""
        let bestServerTimeout = 12000
        

        for(let i=0;i<serverChoices.length;i++){
            if(i<5){
                successHandler()
            }
            try{
                progressHandler("正在测试连接("+i+"/"+serverChoices.length+") : "+serverChoices[i])
                let startTimestamp = Date.now()
                await axios({
                    url:serverChoices[i]+"/handShake",
                    method:'GET',
                    timeout: 5000
                })
                bestServer = serverChoices[i]
                bestServerTimeout = Date.now() - startTimestamp
            }catch(err){

            }
        }
        if(serverChoices.length<5){
            for(let i=0;i<5-serverChoices.length;i++){
                successHandler()
            }
        }
        if(bestServer===""){
            progressHandler("目前无可用的服务器，至"+serverChoices.length+"个服务器的连接均超时")
            throw new Exception("目前无可用的服务器")
        }else{
            localStorage.setItem("altina_rapi_server",bestServer)
        }
    }
    
    async rosFontLoading(successHandler,progressHandler,errorHandler){
        let fontList = [
            ["Novecento",'fonts/Geometos.ttf'],
            ['Geometos','fonts/Geometos.ttf'],
            ['SourceHan','fonts/SourceHanSansCN-Normal.otf'],
            ['SourceHanSerif','fonts/SourceHanSerifSC-Regular.otf'],
            ['SourceHanSerifHeavy','fonts/SourceHanSerifSC-Heavy.otf'],
            ['Bender','fonts/Bender.932867e7.ttf']
        ]
        for(let i=0;i<fontList.length;i++){
            progressHandler("正在加载字体 - "+fontList[i][0])
            let fontFace = new FontFace(fontList[i][0],"url('"+fontList[i][1]+"')")
            await fontFace.load();
	        document.fonts.add(fontFace);
            successHandler()
        }
    }
    rosFontLoaderMonitor(successHandler,progressHandler,errorHandler){
        return new Promise((resolve,reject)=>{
            console.log(document.fonts)
            document.fonts.ready.then(resolve()).catch(reject())
        })
    }
    async rosServerListing(successHandler,progressHandler,errorHandler){
        await new Promise((resolve)=>{
            setTimeout(()=>{
                resolve()
            },1000)
        })
        await axios(
            {
                url:"https://aeroraven.github.io/altina-backend/server_list.json",
                method:'GET',
            }
        ).then((response)=>{
            console.log(response)
            console.log(this)
            Rosemary.serverLists = response.data

        }).catch((error)=>{
            errorHandler("无法连接到aeroraven.github.io，请检查您的代理配置")
        })
    }
    async rosExplorerCheck(){

    }
}