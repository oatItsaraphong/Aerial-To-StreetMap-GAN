import React, { useEffect} from 'react';
import { MapContainer, TileLayer,  useMap,  useMapEvents} from 'react-leaflet';
import './App.css';
import 'leaflet/dist/leaflet.css';
import config from './config.json';

const BASEURL = config.nodeServer;

function Follower({center}) {
  const map = useMap()
  useEffect(()=> {
    if(center){
      map.setView(center,16);
    }
  }, [center, map])
  
  return null
}

function Leader({setCenterShare}) {
  const map = useMapEvents({
    dragend: () => {
      let c = map.getCenter()
      setCenterShare(c)
    }
  })
  return null
}


const MapRequest = ({setCenter, centerShare, mapRef, startArray: defaultCenter, detail, updateValue}) => {
  const version = detail.split(":")[1]
  const type = detail.split(":")[0]
  const url = `${BASEURL}?type=${type}&x={x}&y={y}&z=16&v=${version}`
  return  <MapContainer 
                ref={mapRef} 
                center={defaultCenter} 
                zoom={16} 
                scrollWheelZoom={false} 
                zoomControl={false} 
                style={{width:"100%", height:"80vh"}} 
                doubleClickZoom={false} 
                key={updateValue}>
            <TileLayer url={url} attribution="&copy; <a href=&quot;https://www.openstreetmap.org/copyright&quot;>OpenStreetMap</a> contributors" />
            <Follower center={centerShare}/>
            <Leader setCenterShare={setCenter} />
          </MapContainer>;
}

export default MapRequest;
