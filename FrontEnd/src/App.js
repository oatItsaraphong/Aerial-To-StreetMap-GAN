import React, { useRef, useState } from 'react';
import './App.css';
import 'leaflet/dist/leaflet.css';
import { Container, Row, Col} from 'react-grid-system';
import MapContainerBuild from './MapContainer';
import MapPickerSelector from './MapPicker';
import CityPickerSelector from './CityPicker';
import config from './config.json';
import csuf from './csuflogo.png'

// NY
const city = config.defaultCity;
const title = config.title;

const selectLabelStyle = {fontSize:"20px",  paddingTop:"7px", textAlign:"right"};
const colStyle = {paddingRight:"12px", paddingLeft:"2px"};


function App() {
  console.log(city)
  let defaultCity = [city.fullerton.lat, city.fullerton.long]
  const [centerShare, setCenterShare] = useState();
  const [mapDetailA, mapDetailSetA] = useState('aerial:0');
  const [mapDetailB, mapDetailSetB] = useState('street:0');
  //const [cityCenter, cityCenterSet] = useState([city.ny.lat, city.ny.long]);
  const mapRefA = useRef();
  const mapRefB = useRef();
  return (
    <div className="App" >
      <Container fluid style={{width:"100%", height:"100%"}}>
        <Row style={{height:"15vh"}}>
          <Col md={12}>
            <Row style={{height:"10vh", backgroundColor:"#0f0340", color:"white"}}>
              <Col md={2} style={{textAlign:"right"}}>
                <img height="80px" src={csuf} alt="Logo" />
              </Col>
              <Col>
                <div style={{fontSize:"30px", paddingTop:"20px"}}> 
                  {title}
                </div>
              </Col>
            </Row>

            <Row style={{padding:"4px", height:"4vh"}}>
              <Col md={0.5} style={colStyle}>
                <div style={selectLabelStyle}> City: </div>
              </Col>
              <Col md={2} style={colStyle}>
                <div> <CityPickerSelector setCenter={setCenterShare}/></div>
              </Col>
              <Col md={1.5} style={colStyle}>
                <div style={selectLabelStyle}> Map Type:</div>
              </Col>
              <Col md={2} style={colStyle}>
                <div> <MapPickerSelector effectSetter={mapDetailSetA}/></div>
              </Col>
              <Col md={2.5}>
                <div> </div>
              </Col>
              <Col md={1.5} style={colStyle}>
                <div style={selectLabelStyle}> Map Type:</div>
              </Col>
              <Col md={2} style={colStyle}>
                <div> <MapPickerSelector effectSetter={mapDetailSetB}/></div>
              </Col>
              
            </Row>
          </Col>
        </Row>
        <Row style={{padding:"5px"}}>
          <Col md={6} style={colStyle}>
            <div>
            <MapContainerBuild 
              setCenter={setCenterShare} 
              centerShare={centerShare} 
              mapRef={mapRefA} 
              startArray={defaultCity} 
              detail={mapDetailA} 
              updateValue={mapDetailA} 
            />
            </div>
          </Col>
          <Col md={6} style={colStyle}>
            <div>
            <MapContainerBuild 
              setCenter={setCenterShare} 
              centerShare={centerShare} 
              mapRef={mapRefB} 
              startArray={defaultCity} 
              detail={mapDetailB} 
              updateValue={mapDetailB}
            />
            </div>
          </Col>
        </Row>
        <Row style={{height:"4vh", backgroundColor:"#0f0340", color:"white"}}>
          <Col md={6}>
            </Col>
          <Col md={6}>
            <div style={{fontSize:"12px",  paddingTop:"5px", textAlign:"right"}}> 
                By: Itsaraphong Sawangsri
            </div>
          </Col>
        </Row>
      </Container>

    </div>
  );
}

export default App;
