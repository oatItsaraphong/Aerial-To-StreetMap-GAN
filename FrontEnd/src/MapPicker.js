import React from "react";
import Select from 'react-select'
import config from './config.json';

const options = config.mapTypeValue;


const picker = ({effectSetter}) => {
  let setter = effectSetter;

  const valueChange = (e) => {
    setter(e.value)
  };
  
  return (
      <Select 
        label="Map Type"
        options={options} 
        menuPortalTarget={document.body} 
        styles={{ menuPortal: base => ({ ...base, zIndex: 9999 })}}
        onChange={valueChange}
      />
  );
}

export default picker;
