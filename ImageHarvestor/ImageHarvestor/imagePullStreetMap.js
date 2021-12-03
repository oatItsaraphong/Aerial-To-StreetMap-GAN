const config = require("./config.json");
const citiesJson = require("./cityID.json");
const fs = require("fs");
const request = require('request');
const { resolve } = require("path");


let cities = citiesJson.cities16
let STREET = config.url.streetMap;
let AERIALLIIST = config.url.satMap;
let ZOOM = config.zoomLevel;

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
    return quad.join('');
}


const pickCity = function(){
    return new Promise((resolve, reject)=> {
        const numCities = cities.length;
        let random = Math.floor(Math.random() * numCities) + 0
        random = Math.floor(random);
        resolve(cities[random]);
    });
}

const getLocation = async function(){

    // this to fix
    let city = await pickCity()

    let X = city.x + Math.floor(Math.random() * city.xMax) + 0
    let Y = city.y + Math.floor(Math.random() * city.yMax) + 0

    //image look up

    return {
        cood: {
            x: X,
            y: Y,
        },
        cftid: coodToID(X,Y, ZOOM)
    }
}

const requestImage = function(location, urlIn, filepath = "image" ,convert = false){
    let filename = `${filepath}/${location.cftid}.png`;
    return new Promise((resolve, reject) => {
        let url = urlIn.replace("<zoom>", ZOOM).replace("<x>", location.cood.x).replace("<y>", location.cood.y);
        //console.log(url);
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

async function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}

const main = async (limit) => {



    while(limit > 0){
        let k = await requestImage( await getLocation(),STREET,config.destinationFile.streetMap);
        //console.log(k);
        //await sleep(1000);
        limit--;
    }
}


var myArgs = process.argv.slice(2);
console.log(myArgs)
main(myArgs[0]);

