import React from "react";
import Select from 'react-select';
import config from './config.json';

const options = config.cityCoord;

const picker = ({setCenter}) => {
  let centerSet = setCenter

  const valueChange = (e) => {
    centerSet(e.value)
  };
  
  return (
      <Select 
        options={options} 
        menuPortalTarget={document.body} 
        styles={{ menuPortal: base => ({ ...base, zIndex: 9999 })}}
        onChange={valueChange}
      />

  );
}

export default picker;
