const config = require("./config.json");
const cities = require("./cityID.json");
const fs = require("fs");
const request = require('request');
const { resolve } = require("path");
const { stringify } = require("querystring");
//var globby = require("globby");
//const fs = require('fs');

let STREET = config.url.streetMap;
let AERIALLIIST = config.url.satMap;


//let ZOOM = config.zoomLevel;

const coodToID = function(tileX,tileY, zoom) {
    return new Promise((resolve, reject)=> {
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
        resolve(quad.join(''));
    });
}

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

    //return {x: tileX, y: tileY, zoom: levelOfDetail}
}


// const pickCity = function(){
//     return new Promise((resolve, reject)=> {
//         const numCities = cities.cities.length;
//         let random = Math.floor(Math.random() * numCities) + 0
//         random = Math.floor(random);
//         resolve(cities.cities[random]);
//     });
// }

// const getID = async function(){

//     // this to fix
//     let city = await pickCity()

//     let X = city.x + Math.floor(Math.random() * city.xMax) + 0
//     let Y = city.y + Math.floor(Math.random() * city.yMax) + 0

//     //image look up

//     return {
//         cood: {
//             x: X,
//             y: Y,
//         },
//         cftid: coodToID(X,Y, ZOOM)
//     }
// }

const requestImage = function(location, urlIn, filepath = "image" ,convert = false){

    let filename = `${filepath}/${location.cftid}.png`;
    return new Promise((resolve, reject) => {
        let url = "";
        if(urlIn.includes("<cftid>")){
            url = urlIn.replace("<cftid>", location.cftid)
        }
        else{
         url = urlIn.replace("<zoom>", location.zoom).replace("<x>", location.x).replace("<y>", location.y);
        }
        let option = {
            url: url,
            timeout: 16000
        }

        request.head(option, function(err, res, body){   
            request(option)
            .pipe(fs.createWriteStream(filename))
            .on('close', () => {
                resolve("Done")
            })
            .on('error', () => {
                resolve("Erroe")
            });
        });
    });
}


const getDetail = async function(imageID) {
    let cord = await IdToCood(imageID);
    return {
        cftid: imageID,
        ...cord
    }
    
}


async function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}

const getFileList = (fileName, fileType) =>  {
    const fileList =  fs.readdirSync(fileName);
    return fileList.map(k => k.replace(fileType,""));
}

const main = async (max, startIndex = 0) => {
    
    let outfolder = config.destinationFile.satellitMap;
    let infolder = config.destinationFile.streetMap;

    const fileList =  getFileList(infolder, ".png");
    let index = startIndex;
    max = startIndex + max;


    while(index < max){
        let startIndex = 1;
        let mapSelectionIndex = Math.floor(Math.random() * 3) + 0;
        let imageObj = await getDetail(fileList[index]);

        let k = await requestImage(imageObj, AERIALLIIST[mapSelectionIndex], outfolder);
        //console.log(k);
        //await sleep(1000);
        index++;
        if(index > fileList.length){
            console.log("end of file");
            break;
        }
    }
}


var myArgs = process.argv.slice(2);
main(myArgs[0], myArgs[1]);

