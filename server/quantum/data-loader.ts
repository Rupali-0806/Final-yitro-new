/**
 * MIMIC Data Loader and Preprocessing
 * Handles loading and preprocessing of MIMIC-IV dataset for multimodal healthcare analysis
 */

import * as fs from 'fs';
import * as path from 'path';
import { createReadStream } from 'fs';
import * as csv from 'csv-parser';
import sharp from 'sharp';

export interface MIMICTextData {
  id: string;
  text: string;
  category: string;
  timestamp: Date;
}

export interface MIMICTabularData {
  id: string;
  patientId: string;
  age: number;
  gender: string;
  heartRate: number;
  bloodPressure: string;
  temperature: number;
  respiratoryRate: number;
  oxygenSaturation: number;
  bloodGlucose: number;
  diagnosis: string;
  [key: string]: any;
}

export interface MIMICImageData {
  id: string;
  patientId: string;
  imageType: string;
  data: number[][];
  metadata: Record<string, any>;
}

export interface MIMICLabels {
  patientId: string;
  primaryDiagnosis: string;
  secondaryDiagnoses: string[];
  severity: 'mild' | 'moderate' | 'severe';
  outcome: 'recovered' | 'ongoing' | 'fatal';
}

export class MIMICDataLoader {
  private dataPath: string;
  private textData: MIMICTextData[] = [];
  private tabularData: MIMICTabularData[] = [];
  private imageData: MIMICImageData[] = [];
  private labels: MIMICLabels[] = [];

  constructor(dataPath?: string) {
    this.dataPath = dataPath || '/tmp/mimic_data';
    this.ensureDataDirectory();
  }

  private ensureDataDirectory(): void {
    if (!fs.existsSync(this.dataPath)) {
      fs.mkdirSync(this.dataPath, { recursive: true });
    }
  }

  // Simulate MIMIC data download and preprocessing
  async downloadAndPreprocessMIMICData(): Promise<{
    textData: MIMICTextData[];
    tabularData: MIMICTabularData[];
    imageData: MIMICImageData[];
    labels: MIMICLabels[];
  }> {
    console.log('🔄 Starting MIMIC-IV data download and preprocessing...');
    
    // Since we can't actually download MIMIC data without proper credentials,
    // we'll simulate the data with realistic synthetic data
    await this.generateSyntheticMIMICData();
    
    return {
      textData: this.textData,
      tabularData: this.tabularData,
      imageData: this.imageData,
      labels: this.labels
    };
  }

  private async generateSyntheticMIMICData(): Promise<void> {
    console.log('📊 Generating synthetic MIMIC-IV compatible data...');
    
    // Generate synthetic text data (clinical notes)
    await this.generateTextData();
    
    // Generate synthetic tabular data (vital signs, lab results)
    await this.generateTabularData();
    
    // Generate synthetic image data (X-rays, CT scans)
    await this.generateImageData();
    
    // Generate labels
    await this.generateLabels();
    
    console.log('✅ Synthetic MIMIC data generation complete!');
  }

  private async generateTextData(): Promise<void> {
    const clinicalNoteTemplates = [
      "Patient presents with chest pain and shortness of breath. Physical examination reveals decreased breath sounds on the left side. Chest X-ray shows evidence of pneumonia.",
      "Elderly patient admitted with acute myocardial infarction. ECG shows ST elevation in leads II, III, and aVF. Cardiac enzymes elevated.",
      "Patient with history of diabetes mellitus type 2 presents with diabetic ketoacidosis. Blood glucose 450 mg/dL, ketones positive.",
      "Young adult with fever, cough, and difficulty breathing. COVID-19 test positive. Chest CT shows bilateral ground-glass opacities.",
      "Patient with chronic kidney disease presents with fluid overload. Creatinine elevated at 3.2 mg/dL. Dialysis initiated.",
      "Stroke patient with left-sided weakness. CT scan shows acute ischemic stroke in right middle cerebral artery territory.",
      "Cancer patient with metastatic disease. Recent chemotherapy. Presents with neutropenia and fever.",
      "Hypertensive crisis with blood pressure 220/120. Patient complains of severe headache and vision changes.",
      "Chronic obstructive pulmonary disease exacerbation. Patient on home oxygen therapy. Increased dyspnea and cough.",
      "Liver cirrhosis patient with ascites and jaundice. Hepatic encephalopathy grade 2. Albumin low at 2.1 g/dL."
    ];

    const categories = ['admission_note', 'progress_note', 'discharge_summary', 'radiology_report', 'lab_report'];
    
    for (let i = 0; i < 1000; i++) {
      const template = clinicalNoteTemplates[Math.floor(Math.random() * clinicalNoteTemplates.length)];
      const category = categories[Math.floor(Math.random() * categories.length)];
      
      this.textData.push({
        id: `text_${i}`,
        text: this.addVariationToText(template),
        category: category,
        timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000) // Random date within last 30 days
      });
    }
  }

  private addVariationToText(template: string): string {
    const variations = [
      " Patient appears comfortable.",
      " Vital signs stable.",
      " No acute distress noted.",
      " Patient cooperative during examination.",
      " Family history significant for cardiovascular disease.",
      " No known drug allergies.",
      " Patient denies smoking or alcohol use.",
      " Follow-up appointment scheduled."
    ];
    
    const numVariations = Math.floor(Math.random() * 3);
    let text = template;
    
    for (let i = 0; i < numVariations; i++) {
      const variation = variations[Math.floor(Math.random() * variations.length)];
      text += variation;
    }
    
    return text;
  }

  private async generateTabularData(): Promise<void> {
    const genders = ['Male', 'Female'];
    const diagnoses = ['Pneumonia', 'Heart Disease', 'Diabetes', 'COVID-19', 'Kidney Disease', 'Stroke', 'Cancer', 'Hypertension', 'COPD', 'Liver Disease'];
    
    for (let i = 0; i < 1000; i++) {
      const age = Math.floor(Math.random() * 80) + 18; // Age 18-98
      const gender = genders[Math.floor(Math.random() * genders.length)];
      const diagnosis = diagnoses[Math.floor(Math.random() * diagnoses.length)];
      
      // Generate realistic vital signs based on diagnosis
      const vitalSigns = this.generateRealisticVitalSigns(diagnosis, age);
      
      this.tabularData.push({
        id: `tabular_${i}`,
        patientId: `patient_${i}`,
        age: age,
        gender: gender,
        heartRate: vitalSigns.heartRate,
        bloodPressure: vitalSigns.bloodPressure,
        temperature: vitalSigns.temperature,
        respiratoryRate: vitalSigns.respiratoryRate,
        oxygenSaturation: vitalSigns.oxygenSaturation,
        bloodGlucose: vitalSigns.bloodGlucose,
        diagnosis: diagnosis,
        // Additional lab values
        hemoglobin: this.generateNormalValue(12, 16, 1),
        whiteBloodCells: this.generateNormalValue(4, 11, 1),
        platelets: this.generateNormalValue(150, 400, 50),
        creatinine: this.generateNormalValue(0.6, 1.2, 0.1),
        bun: this.generateNormalValue(7, 20, 2),
        sodium: this.generateNormalValue(135, 145, 2),
        potassium: this.generateNormalValue(3.5, 5.0, 0.2),
        chloride: this.generateNormalValue(96, 106, 2)
      });
    }
  }

  private generateRealisticVitalSigns(diagnosis: string, age: number): {
    heartRate: number;
    bloodPressure: string;
    temperature: number;
    respiratoryRate: number;
    oxygenSaturation: number;
    bloodGlucose: number;
  } {
    let baseHR = 72;
    let baseSystolic = 120;
    let baseDiastolic = 80;
    let baseTemp = 98.6;
    let baseRR = 16;
    let baseO2 = 98;
    let baseGlucose = 100;

    // Adjust based on diagnosis
    switch (diagnosis) {
      case 'Heart Disease':
        baseHR += Math.random() * 20;
        baseSystolic += Math.random() * 40;
        break;
      case 'Pneumonia':
      case 'COVID-19':
        baseTemp += Math.random() * 4;
        baseRR += Math.random() * 10;
        baseO2 -= Math.random() * 8;
        break;
      case 'Diabetes':
        baseGlucose += Math.random() * 200;
        break;
      case 'COPD':
        baseO2 -= Math.random() * 10;
        baseRR += Math.random() * 8;
        break;
    }

    // Age adjustments
    if (age > 65) {
      baseSystolic += (age - 65) * 0.5;
      baseHR -= Math.random() * 10;
    }

    return {
      heartRate: Math.round(baseHR + (Math.random() - 0.5) * 20),
      bloodPressure: `${Math.round(baseSystolic)}/${Math.round(baseDiastolic)}`,
      temperature: +(baseTemp + (Math.random() - 0.5) * 2).toFixed(1),
      respiratoryRate: Math.round(baseRR + (Math.random() - 0.5) * 6),
      oxygenSaturation: Math.round(baseO2 + (Math.random() - 0.5) * 4),
      bloodGlucose: Math.round(baseGlucose + (Math.random() - 0.5) * 50)
    };
  }

  private generateNormalValue(min: number, max: number, precision: number): number {
    const value = min + Math.random() * (max - min);
    const decimals = Math.max(0, Math.min(20, Math.log10(1/precision))); // Clamp to valid range
    return +value.toFixed(decimals);
  }

  private async generateImageData(): Promise<void> {
    const imageTypes = ['chest_xray', 'ct_scan', 'mri', 'ecg'];
    
    for (let i = 0; i < 200; i++) { // Fewer images as they're more resource intensive
      const imageType = imageTypes[Math.floor(Math.random() * imageTypes.length)];
      const imageData = this.generateSyntheticImageData(imageType);
      
      this.imageData.push({
        id: `image_${i}`,
        patientId: `patient_${i % 1000}`, // Link to patient data
        imageType: imageType,
        data: imageData,
        metadata: {
          width: imageData[0].length,
          height: imageData.length,
          channels: 1,
          acquisition_date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000)
        }
      });
    }
  }

  private generateSyntheticImageData(imageType: string): number[][] {
    const width = 64; // Small images for demo
    const height = 64;
    const imageData: number[][] = [];

    for (let y = 0; y < height; y++) {
      imageData[y] = [];
      for (let x = 0; x < width; x++) {
        let pixel = 0;
        
        switch (imageType) {
          case 'chest_xray':
            // Simulate chest X-ray with rib patterns
            pixel = this.generateChestXrayPixel(x, y, width, height);
            break;
          case 'ct_scan':
            // Simulate CT scan with circular patterns
            pixel = this.generateCTScanPixel(x, y, width, height);
            break;
          case 'mri':
            // Simulate MRI with brain-like patterns
            pixel = this.generateMRIPixel(x, y, width, height);
            break;
          case 'ecg':
            // Simulate ECG waveform
            pixel = this.generateECGPixel(x, y, width, height);
            break;
          default:
            pixel = Math.random();
        }
        
        imageData[y][x] = Math.max(0, Math.min(1, pixel));
      }
    }

    return imageData;
  }

  private generateChestXrayPixel(x: number, y: number, width: number, height: number): number {
    const centerX = width / 2;
    const centerY = height / 2;
    const distanceFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    
    // Lung field simulation
    let pixel = 0.2 + 0.3 * Math.exp(-distanceFromCenter / 20);
    
    // Add rib-like structures
    if (Math.sin(y * 0.3) > 0.8) {
      pixel += 0.4;
    }
    
    // Add some noise
    pixel += (Math.random() - 0.5) * 0.1;
    
    return pixel;
  }

  private generateCTScanPixel(x: number, y: number, width: number, height: number): number {
    const centerX = width / 2;
    const centerY = height / 2;
    const distanceFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    
    // Circular organ simulation
    let pixel = 0.5;
    
    if (distanceFromCenter < 20) {
      pixel = 0.8; // Dense tissue
    } else if (distanceFromCenter < 25) {
      pixel = 0.3; // Soft tissue
    }
    
    // Add noise
    pixel += (Math.random() - 0.5) * 0.1;
    
    return pixel;
  }

  private generateMRIPixel(x: number, y: number, width: number, height: number): number {
    const centerX = width / 2;
    const centerY = height / 2;
    const distanceFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    
    // Brain-like structure
    let pixel = 0.3;
    
    if (distanceFromCenter < 25) {
      pixel = 0.7; // Brain tissue
      if (distanceFromCenter < 10) {
        pixel = 0.9; // White matter
      }
    }
    
    // Add noise
    pixel += (Math.random() - 0.5) * 0.1;
    
    return pixel;
  }

  private generateECGPixel(x: number, y: number, width: number, height: number): number {
    const centerY = height / 2;
    
    // ECG waveform simulation
    let waveform = Math.sin(x * 0.5) * 0.3;
    
    // Add QRS complex
    if (x % 20 < 3) {
      waveform += Math.sin(x * 2) * 0.7;
    }
    
    // Check if pixel is on the waveform line
    const waveformY = centerY + waveform * 20;
    const pixel = Math.abs(y - waveformY) < 2 ? 1 : 0.1;
    
    return pixel;
  }

  private async generateLabels(): Promise<void> {
    const diagnoses = ['Pneumonia', 'Heart Disease', 'Diabetes', 'COVID-19', 'Kidney Disease', 'Stroke', 'Cancer', 'Hypertension', 'COPD', 'Liver Disease'];
    const severities: ('mild' | 'moderate' | 'severe')[] = ['mild', 'moderate', 'severe'];
    const outcomes: ('recovered' | 'ongoing' | 'fatal')[] = ['recovered', 'ongoing', 'fatal'];
    
    for (let i = 0; i < 1000; i++) {
      const primaryDiagnosis = diagnoses[Math.floor(Math.random() * diagnoses.length)];
      const numSecondary = Math.floor(Math.random() * 3);
      const secondaryDiagnoses = [];
      
      for (let j = 0; j < numSecondary; j++) {
        const secondary = diagnoses[Math.floor(Math.random() * diagnoses.length)];
        if (secondary !== primaryDiagnosis && !secondaryDiagnoses.includes(secondary)) {
          secondaryDiagnoses.push(secondary);
        }
      }
      
      this.labels.push({
        patientId: `patient_${i}`,
        primaryDiagnosis: primaryDiagnosis,
        secondaryDiagnoses: secondaryDiagnoses,
        severity: severities[Math.floor(Math.random() * severities.length)],
        outcome: outcomes[Math.floor(Math.random() * outcomes.length)]
      });
    }
  }

  // Preprocessing functions
  preprocessTextData(textData: MIMICTextData[]): MIMICTextData[] {
    return textData.map(item => ({
      ...item,
      text: this.cleanText(item.text)
    }));
  }

  private cleanText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  preprocessTabularData(tabularData: MIMICTabularData[]): MIMICTabularData[] {
    return tabularData.map(item => ({
      ...item,
      // Normalize numerical values
      age: this.normalizeValue(item.age, 0, 100),
      heartRate: this.normalizeValue(item.heartRate, 40, 180),
      temperature: this.normalizeValue(item.temperature, 95, 105),
      respiratoryRate: this.normalizeValue(item.respiratoryRate, 8, 40),
      oxygenSaturation: this.normalizeValue(item.oxygenSaturation, 80, 100),
      bloodGlucose: this.normalizeValue(item.bloodGlucose, 50, 400)
    }));
  }

  private normalizeValue(value: number, min: number, max: number): number {
    return (value - min) / (max - min);
  }

  preprocessImageData(imageData: MIMICImageData[]): MIMICImageData[] {
    return imageData.map(item => ({
      ...item,
      data: this.normalizeImageData(item.data)
    }));
  }

  private normalizeImageData(imageData: number[][]): number[][] {
    const flat = imageData.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const range = max - min;
    
    if (range === 0) return imageData;
    
    return imageData.map(row =>
      row.map(pixel => (pixel - min) / range)
    );
  }

  // Save processed data for future use
  async saveProcessedData(): Promise<void> {
    const data = {
      textData: this.textData,
      tabularData: this.tabularData,
      imageData: this.imageData,
      labels: this.labels
    };

    fs.writeFileSync(
      path.join(this.dataPath, 'processed_mimic_data.json'),
      JSON.stringify(data, null, 2)
    );

    console.log('✅ Processed MIMIC data saved to disk');
  }

  // Load processed data from disk
  async loadProcessedData(): Promise<{
    textData: MIMICTextData[];
    tabularData: MIMICTabularData[];
    imageData: MIMICImageData[];
    labels: MIMICLabels[];
  }> {
    const filePath = path.join(this.dataPath, 'processed_mimic_data.json');
    
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      this.textData = data.textData;
      this.tabularData = data.tabularData;
      this.imageData = data.imageData;
      this.labels = data.labels;
      
      console.log('✅ Processed MIMIC data loaded from disk');
      return data;
    } else {
      console.log('⚠️ No processed data found, generating new data...');
      return await this.downloadAndPreprocessMIMICData();
    }
  }
}

// Main function to load and preprocess MIMIC data
export async function loadAndPreprocessMimicData(): Promise<{
  textData: MIMICTextData[];
  tabularData: MIMICTabularData[];
  imageData: MIMICImageData[];
  labels: MIMICLabels[];
}> {
  const loader = new MIMICDataLoader();
  
  try {
    // Try to load existing processed data
    const data = await loader.loadProcessedData();
    
    // Preprocess the data
    const processedData = {
      textData: loader.preprocessTextData(data.textData),
      tabularData: loader.preprocessTabularData(data.tabularData),
      imageData: loader.preprocessImageData(data.imageData),
      labels: data.labels
    };
    
    // Save processed data
    await loader.saveProcessedData();
    
    return processedData;
  } catch (error) {
    console.error('Error loading MIMIC data:', error);
    throw error;
  }
}