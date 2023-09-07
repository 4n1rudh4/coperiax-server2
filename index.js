const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const Papa = require('papaparse');
const fs = require('fs');
const cors = require('cors'); // Add CORS middleware

const app = express();
const port = process.env.PORT || 3000;

let model;
app.use(cors()); 

async function loadModel() {
  model = await tf.loadLayersModel('./model.json');
  console.log('Model loaded successfully.');
}

loadModel().catch(error => {
  console.error('Error loading model:', error);
});

app.use(bodyParser.json());

app.post('/predict', (req, res) => {
    if (!model) {
      return res.status(500).json({ error: 'Model not loaded yet' });
    }
  
    try {
    const inputData = {"Temperature":27,"Humidity":47,"Moisture":26,"Soil_Type":'Sandy',"Crop_Type":'Maize',"Nitrogen":37,"Potassium":0,"Phosphorous":0};
  
     
    const numSoilTypes = getNumSoilTypes(); 
    const numCropTypes = getNumCropTypes();
    function getNumSoilTypes() {
        const dataset = Papa.parse(fs.readFileSync('data2.csv', 'utf8'), {
          header: true, 
          dynamicTyping: true, 
        }).data;
      
        const soilTypes = [...new Set(dataset.map(row => row['Soil_Type']))];
      
        return soilTypes.length;
      }
      
      function getNumCropTypes() {
        const dataset = Papa.parse(fs.readFileSync('data2.csv', 'utf8'), {
          header: true, 
          dynamicTyping: true, 
        }).data;
      
        const cropTypes = [...new Set(dataset.map(row => row['Crop_Type']))];
      
        return cropTypes.length;
      }
      const soilTypeMapping = {
        'Sandy': 0,   
        'Loamy': 1,
        'Black': 2,
        'Red':3,
        'Clayey':4
      };
      const cropTypeMapping = {
        'Maize': 0,    
        'Sugarcane': 1,
        'Cotton': 2,
        'Tobacco':3,
        'Paddy':4,
        'Barley':5,
        'Wheat':6,
        'Millets':7,
        'Oil seeds':8,
        'Pulses':9,
        'Ground Nuts':10
      };
      
      const soilTypeIndex = soilTypeMapping[inputData.Soil_Type];
     const cropTypeIndex = cropTypeMapping[inputData.Crop_Type];
     console.log(soilTypeIndex);
     console.log(cropTypeIndex);


      const numericalFeatures = tf.tensor2d([
        [
          inputData.Temperature,
          inputData.Humidity,
          inputData.Moisture,
          soilTypeIndex,
          cropTypeIndex,
          inputData.Nitrogen,
          inputData.Potassium,
          inputData.Phosphorous,
        ]]);

      console.log('numericalFeatures shape:', numericalFeatures.shape);
      const fertilizer=['Urea','DAP','14-35-14','28-28','17-17-17','20-20','10-26-26'];
      const predictions = model.predict(numericalFeatures);
      const predictionData = predictions.arraySync();
      const predictedFertilizerIndex = predictionData[0].indexOf(Math.max(...predictionData[0]));
      const predictedFertilizer = fertilizer[predictedFertilizerIndex];
  
      res.json({ predictions: predictedFertilizer });
    } catch (error) {
      console.error('Error making predictions:', error);
      res.status(500).json({ error: 'Error making predictions' });
    }
  });

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
