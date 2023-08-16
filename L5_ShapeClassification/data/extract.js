var fs = require('fs');
var ndjson = require('ndjson'); // npm install ndjson
var arguments = process.argv.splice(2)[0]; //获取传入的label

function parseSimplifiedDrawings(fileName, callback) {
  var drawings = [];
  var fileStream = fs.createReadStream(fileName)
  fileStream
    .pipe(ndjson.parse())
    .on('data', function(obj) {
      drawings.push(obj)
    })
    .on("error", callback)
    .on("end", function() {
      callback(null, drawings)
    });
}

//文件路径修改!!：
let filepath = "./dataset/quickdraw/"+arguments+"/"+arguments

parseSimplifiedDrawings(filepath+".ndjson", function(err, drawings) {
  if(err) return console.error(err);
  drawings.forEach(function(d) {
    // Do something with the drawing
    console.log(d.key_id, d.countrycode);
  })
  console.log("# of drawings:", drawings);
  var filename = filepath+".json";//保存的路径
  fs.writeFileSync(filename, JSON.stringify(drawings));
})