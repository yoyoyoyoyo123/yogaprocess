let video;
let poseNet;
let pose;
let skeleton;

let brain;

let state = 'waiting';
let targetLabel;
let poseLabel;

// // Create WebSocket connection.
// const socket = new WebSocket('ws://localhost:8080');

// // Connection opened
// socket.addEventListener('open', function (event) {
//     socket.send('Hello Server!');
// });

// // Listen for messages
// socket.addEventListener('message', function (event) {
//     console.log('Message from server ', event.data);
// });

    // 建立 WebSocket (本例 server 端於本地運行)
    let url = 'ws://192.168.43.182:3000'
    var ws = new WebSocket(url)
    // 監聽連線狀態
    ws.onopen = () => {
      console.log('open connection')
    }
    ws.onclose = () => {
      console.log('close connection');
    }
    //接收 Server 發送的訊息
    ws.onmessage = event => {
      let txt = event.data
      if (!showDom.value) showDom.value = txt
      else showDom.value = showDom.value + "\n" + txt
      keyinDom.value = ""
    }

    function keyPressed() {
      if(key=='s'){
        brain.saveData();
      } else{
      targetLabel = key;
      console.log(targetLabel);
      setTimeout(function(){
        console.log('collecting');
        state = 'collecting';
        setTimeout(function(){
        console.log('not collecting');
        state = 'waiting';
      },10000);
      },10000);
    }
    }
    
    function setup() {
      createCanvas(640, 480);
      video=createCapture(VIDEO);
      video.hide();
      poseNet = ml5.poseNet(video,modelLoaded);
      poseNet.on('pose',gotPoses);
      
      let options={
        inputs:34,
        outputs:5,
        task:'classification',
        debug:true
      }
      brain = ml5.neuralNetwork(options);
        const modelInfo = {
        model: 'sunyoga1/model.json',
        metadata: 'sunyoga1/model_meta.json',
        weights: 'sunyoga1/model.weights.bin',
      };
      brain.load(modelInfo, brainLoaded);
      //brain.loadData('yca.json',dataReady);
    }
    function brainLoaded(){
      console.log('pose classifiation Ready!');
      classifyPose();
    
    }
    function classifyPose(){
      if(pose){
              let inputs = [];
            for(let i = 0;i < pose.keypoints.length;i++){
          let x = pose.keypoints[i].position.x;
          let y = pose.keypoints[i].position.y;
              inputs.push(x);
              inputs.push(y);
        }
      brain.classify(inputs,gotResult);
      }else{
        setTimeout(classifyPose,100);
      }
    }
    
    function gotResult(error,results){
      //console.log(results);
      if(results[0].confidence > 0.6){
        poseLabel = results[0].label;
      }
      ws.send(poseLabel);
      // console.log(results[0].confidence);
      classifyPose();
    }
    
    function dataReady(){
      brain.normalizeData();
      brain.train({epochs:50},finished);
    }
    function finished(){
      console.log('model trained');
      brain.save();
    }
    
    function gotPoses(poses){
      //console.log(poses);
      if(poses.length>0){
        pose = poses[0].pose;
        skeleton = poses[0].skeleton;
        if(state == 'collecting'){
        let inputs = [];
            for(let i = 0;i < pose.keypoints.length;i++){
          let x = pose.keypoints[i].position.x;
          let y = pose.keypoints[i].position.y;
              inputs.push(x);
              inputs.push(y);
        }
        let target = [targetLabel];
        brain.addData(inputs,target);
      }
    }
    }
    
    function modelLoaded(){
      console.log('poseNet ready');
      
    }
    function draw() {
      // push();
      translate(video.width,0);
      scale(-1,1);
      image(video,0,0,video.width,video.height);
      if(pose){
        fill(255,0,0);
        ellipse(pose.nose.x,pose.nose.y,32);
        fill(0,0,255);
        ellipse(pose.rightWrist.x,pose.rightWrist.y,32);
        ellipse(pose.leftWrist.x,pose.leftWrist.y,32);
        
        for(let i = 0;i < pose.keypoints.length;i++){
          let x = pose.keypoints[i].position.x;
          let y = pose.keypoints[i].position.y;
          fill(0,255,0);
          ellipse(x,y,16,16);
        }
        for(let i = 0;i < skeleton.length;i++){
          let a = skeleton[i][0];
          let b = skeleton[i][1];
          strokeWeight(2);
          stroke(255);
          line(a.position.x,a.position.y,b.position.x,b.position.y);
        }
      }
      // pop();
      // fill(255,0,255);
      // noStroke();
      // textSize(256);
      // textAlign(CENTER,CENTER);
      // text(poseLabel,width/2,height/2);
      
    }