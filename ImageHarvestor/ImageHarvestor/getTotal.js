const citiesSize = require("./cityID.json");


let total15 = 0;

for(let i = 0; i < citiesSize.cities.length; i++){
    total15 = total15 + (citiesSize.cities[i].xMax * citiesSize.cities[i].yMax)
}

console.log("Total 15:", total15);
let total16 = 0;

for(let i = 0; i < citiesSize.cities16.length; i++){
    total16 = total16 + (citiesSize.cities16[i].xMax * citiesSize.cities16[i].yMax)
}

console.log("Total 16:", total16);