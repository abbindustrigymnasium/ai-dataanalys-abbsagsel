function setup() {
  createCanvas(400, 400);

}
var b = 0;
var m = 0;
var data = [];

function mousePressed() {
  var x = map(mouseX, 0, width, 0, 1);
  var y = map(mouseY, 0, height, 1, 0);
  var point = createVector(x, y);
  data.push(point);
}

function gradientDescent() {
  var learning_rate = 0.01;//hur snabbt linjen anpassar sig
  
  for (var i = 0; i < data.length; i++){
    var x = data[i].x;
    var y = data[i].y;
    
    var guess = m * x + b;
    
    var error = y - guess;
    
    m += (error * x) * learning_rate;
    b += (error) * learning_rate;
  }
}

function drawLine() {
  //x1 är början på x, x2 slutet. Y1 och y2 bestäms utifrån m*x+b
  var x1 = 0;
  var y1 = m * x1 + b;
  var x2 = 1;
  var y2 = m * x2 + b;

  x1 = map(x1, 0, 1, 0, width);
  y1 = map(y1, 0, 1, height, 0);
  x2 = map(x2, 0, 1, 0, width);
  y2 = map(y2, 0, 1, height, 0);

  stroke(255, 0, 255);
  line(x1, y1, x2, y2);

}

function draw() {
  background(51);
  
    //Rita ut varje datapunkt
  for (let i = 0; i < data.length; i++) {
    var x = map(data[i].x, 0, 1, 0, width);
    var y = map(data[i].y, 0, 1, height, 0);
    fill(255);
    stroke(255);
    ellipse(x, y, 8, 8)

  }
  //Om vi har mer än 1 punkt räkna ut och rita ut sträcket.
  if (data.length > 1) {
    gradientDescent();
    drawLine();
  }

}