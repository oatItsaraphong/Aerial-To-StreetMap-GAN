const zerorpc = require('zerorpc');

const express = require('express')
const app = express()
const fs = require('fs')
//const request = require('request');
const request = require('request');
const pathLib = require('path');


//const 


process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = 0


const config = {
    cache_path: "./cacheimages",
    path: {
      aerial: "./cacheimages/aerialimage",
      aerialread: "./cacheimages/aerialimageread",
      streetgen: "./cacheimages/streetgen",
      streetreal: "./cacheimages/streetreal"
    },
    nodeserver: {
      port: 3001,
    },
    genserver: {
      url: {
        15: 4015,
        28: 4028,
        16: 4016
      },
      base: "tcp://127.0.0.1:",
      genFunciton: "generate_image"
    },
    map: {
      aerial: "https://ecn.t1.tiles.virtualearth.net/tiles/a<cftid>.jpeg?g=10993",
      street: "https://tiles.wmflabs.org/osm-no-labels/<zoom>/<x>/<y>.png"
    },
    key: {
      aerial: "AERIAL",
      street: "STREET",
      gan: "GAN"
    },
    zoomlevel: 16

}


const cftidRegex = new RegExp('^[0123]*$')
let port = config.nodeserver.port;


// sample call to python generator
app.get('/generate', (req, res) => {
    res.send('Hello World!')
    var client = new zerorpc.Client();
    client.connect(config.genserver.url);
    client.invoke(config.genserver.genFunciton, "0212300211313130.png", function(error, res, more) {
        console.log("test");
    });
})

// query for python code to generate the image
async function generateImage(imageName, version){
  return new Promise((resolve, request) => {
    //console.log("GPY:start")
    var client = new zerorpc.Client();
    gen_url = config.genserver.base + String(config.genserver.url[parseInt(version)])
    client.connect(gen_url);
    client.invoke(config.genserver.genFunciton, imageName, function(error, res, more) {
      //console.log("GPY:resolve")
        resolve();
    });
  });
}

//
const clipNum = function(num, min, max) {
  return Math.min(Math.max(num, min), max);
}

const mapSizeFun = function(zoom) {
  return 256 << zoom
}

const latLongToPixel = function(lat, long, zoomlevel){
  
  const EarthRadius = 6378137;  
  const MinLatitude = -85.05112878;  
  const MaxLatitude = 85.05112878;  
  const MinLongitude = -180;  
  const MaxLongitude = 180;  
  let latitude = clipNum(lat, MinLatitude, MaxLatitude);  
  let longitude = clipNum(long, MinLongitude, MaxLongitude);  

  let x = (longitude + 180) / 360;   
  let sinLatitude = Math.sin(latitude * Math.PI / 180);  Math.si
  let y = 0.5 - Math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * Math.PI);  
  let mapSize = mapSizeFun(zoomlevel);  
  let pixelX = clipNum(x * mapSize + 0.5, 0, mapSize - 1) | 0;  
  let pixelY = clipNum(y * mapSize + 0.5, 0, mapSize - 1) | 0;  
  return {x: pixelX, y: pixelY};
}

const pixelToTileXY = function (pixelX, pixelY) {
  return {x: pixelX / 256, y: pixelY /256}
}

const latLongToTileXY = function (lat, long, zoom){
  let pixelSet = latLongToPixel(lat, long, zoom)
  return pixelToTileXY(pixelSet.x, pixelSet.y)
}


// coordinate to cftid
const coodToID = function(tileX,tileY, zoom) {

      let quad = [];
      for(let i = zoom; i > 0; i--){
          let digit = 0;
          let mask = 1 << (i-1)
          if ((tileX & mask) != 0){
              digit += 1 
          }
          if ((tileY & mask) != 0){
              digit+= 1  
              digit+= 1  
          }
          quad.push(digit)
      }
      return(quad.join(''));

}

// translate cftid to coordinate
const IdToCood = function(quadKey) 
{
    return new Promise((resolve, reject)=> {
        let tileX = tileY = 0;  
        let levelOfDetail = quadKey.length;  
        //console.log(quadKey.le)
        for (let i = levelOfDetail; i > 0; i--)  
        {  
            let mask = 1 << (i - 1);  
            
            switch (quadKey[levelOfDetail - i])  
            {  
                case '0':  
                    break;  

                case '1':  
                    tileX |= mask;  
                    break;  

                case '2':  
                    tileY |= mask;  
                    break;  

                case '3':  
                    tileX |= mask;  
                    tileY |= mask;  
                    break;  

                default:  
                    throw new ArgumentException("Invalid QuadKey digit sequence.");  
            }  
        }  
        resolve({x: tileX, y: tileY, zoom: levelOfDetail});
    });
}

// get image from third party
const requestImage = function(locationCftid, urlIn, filename ,convert = false){
  //let filename = `${filepath}/${locationCftid}.png`;
  return new Promise(async (resolve, reject) => {
      let url = "";
      if(urlIn.includes("<cftid>")){
          url = urlIn.replace("<cftid>", locationCftid)
      }
      else{
          location = await IdToCood(locationCftid)
          url = urlIn.replace("<zoom>", location.zoom).replace("<x>", location.x).replace("<y>", location.y);
      }
      let option = {
          url: url,
          timeout: 16000,
          maxAttempts: 5,  
          retryDelay: 5000
      }
      //console.log("request")
      request.head(option, function(err, res, body){   
          request(option)
          .pipe(fs.createWriteStream(filename))
          .on('close', () => {
              //console.log("Done")
              resolve(filename);
          })
          .on('error', () => {
            //console.log("EERR")
              resolve("Erroe")
          });
      });
  });
}

async function waitForFileExists(filePath, currentTime = 0, timeout = 3000) {
  if (fs.existsSync(filePath)) return true;
  if (currentTime === timeout) return false;
  // wait for 1 second
  await new Promise((resolve, reject) => setTimeout(() => resolve(true), 1000));
  // waited for 1 second
  return waitForFileExists(filePath, currentTime + 1000, timeout);
}

// stream the image back
async function pipeImage(path, type, res){
  let retry = 3;
  try{
        let fileexist = await waitForFileExists(path);
        if(fileexist){
          var s = fs.createReadStream(path);
          s.on('open', function () {

              res.set('Content-Type', type);
              res.set('Refresh', 3)
              s.pipe(res);
              //res.sendFile(path);
              
          });
          s.on('error', function (a, b) {
            console.log(a)

          });
        }
        else{
          var a = fs.createReadStream("E:\\project\\webserver\\cacheimages\\aerialimage\\0230132002033130.png");
          //console.log("pipe Fail")
          res.set('Content-Type', type);
          res.set('Refresh', 3)
          res.set('Cache-Control', "max-age=30000")
          a.pipe(res);
        }

  }
  catch(err){
    console.log(err)
    throw err
  }

}

// get map image that is not generated
async function getStreetMap(name,  res){
  let dir = config.path.streetreal;
  let mapServer =  config.map.street;
  let type =  "png";
  let contenttype =  "image/png";
  //console.log("getSteetmap")
  try{
      let path = dir + "/" + name + "."+ type
      if(fs.existsSync(path)){
          pipeImage(path, contenttype, res);
      }
      else{
          //console.log('not cache') 
          await requestImage(name, mapServer, path);
          pipeImage(path, contenttype, res);
      }
  }
  catch(err){
    throw err
  }
}

// get map image that is not generated
async function getAerialMap(name, res){
  let dir = config.path.aerial;
  let mapServer =  config.map.aerial;
  let type =  "png";
  let contenttype =  "image/png";
  try{
      let path = dir + "/" + name + "."+ type;
      if(fs.existsSync(path)){
          //console.log("A:cache")
          pipeImage(path, contenttype, res);
      }
      else{
          //console.log('A:not cache') 
          let outpath = await requestImage(name, mapServer, path);
          //console.log(outpath)
          pipeImage(path, contenttype, res);
      }
  }
  catch(err){
    throw err
  }
}

// get generate map from ML mode
async function getGenerateMap(name, versionIn,res){
  let dir = `${config.path.streetgen}v${versionIn}`
  let version = versionIn
  let type =  "png";
  let contenttype =  "image/png";
  try{
      let path = dir + "/" + name + "."+ type

      if(fs.existsSync(path)){
          //console.log('G:cache')
          pipeImage(path, contenttype, res);
      }
      else
      {
        //console.log('G:not cache')

          //get aerial image if not exist
          let source = config.path.aerial + "/" + name + "."+ type
          if(!fs.existsSync(source)){
            await requestImage(name, config.map.aerial, source);
          }
          let aerialpath = config.path.aerial + "/" + name + "."+ type
          //let fileexist = await waitForFileExists(aerialpath);
          //--// console.log('not cache')
          await generateImage(name + "."+ type, version);
          pipeImage(path, contenttype, res);
      }
  }
  catch(err){
      throw err
  }
}

// main entry point to get map 
app.get('/getmapimage', (req, res) => {
    // check if image in cache (filesystem)
    try{

        //==// console.log(req.query)
        // Parameter check
        if(req.query === null){
          throw new Error('missing params query')
        }

        // if(cftidRegex.test(req.query.cftid)){
        //   throw new Error('Invalid CFTID: data not in correct format')
        // }

        // if(req.query.cftid.length != config.zoomlevel){
        //   throw new Error(`Invalid CFTID length: the ID should be at length: ${config.zoomlevel}`)
        // }

        if(req.query.x === null && req.query.y === null){
          throw new Error('Invalid Coordinate: data not in correct format')
        }

        if(req.query.z != config.zoomlevel){
          throw new Error(`Invalid Zoom level: the ID should be at length: ${config.zoomlevel}`)
        }
      
        if(req.query.v === null){
          throw new Error(`Invalid Version`)
        }

        //const tile = latLongToTileXY(req.query.x, req.query.y, req.query.z)
        const cftid = coodToID(req.query.x, req.query.y, req.query.z);
        const type = req.query.type.toUpperCase()
        //check if file in
        if(type === config.key.aerial){
          getAerialMap(cftid, res);
        }
        else if(type === config.key.street){
          getStreetMap(cftid, res);
        }
        else if(type === config.key.gan){
          getGenerateMap(cftid, req.query.v, res);
        }
        else{
          throw new Error( 'Invalide query');
        }
    }
    catch(err){
        res.send({message: err.message});
    }


})

app.get('/clearcache', (req, res) => {
    
    removeDir = function(dirPath) {
      try { 
        var files = fs.readdirSync(dirPath); 
      }
      catch(e) { 
        return; 
      }
      if (files.length > 0)
        for (var i = 0; i < files.length; i++) 
        {
          var filePath = dirPath + '/' + files[i];
          if (fs.statSync(filePath).isFile())
            fs.unlinkSync(filePath);
          else
          removeDir(filePath);
        }
    }
    removeDir(config.cache_path);
  

    res.send({message: "Remove all cache"});
})

app.get('/health', (req, res) => {
    res.send('Hello World!2')
})


var arguments = process.argv
port = arguments[2]

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
});

