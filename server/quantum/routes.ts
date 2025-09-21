/**
 * Quantum Healthcare API Routes
 * Express routes for the quantum healthcare prediction system
 */

import { Router, Request, Response } from 'express';
import multer from 'multer';
import sharp from 'sharp';
import { QuantumHealthcareModel } from './models';
import { loadAndPreprocessMimicData } from './data-loader';

const router = Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Allow images and text files
    if (file.mimetype.startsWith('image/') || file.mimetype.startsWith('text/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image and text files are allowed'));
    }
  },
});

// Initialize the quantum healthcare model
const quantumModel = new QuantumHealthcareModel();
let isModelTrained = false;

// Health check endpoint
router.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'quantum-healthcare-api',
    timestamp: new Date().toISOString(),
    modelTrained: isModelTrained
  });
});

// Load and train model endpoint
router.post('/train', async (req: Request, res: Response) => {
  try {
    res.json({ 
      message: 'Training started', 
      status: 'training' 
    });

    // Load MIMIC data in background
    console.log('🔄 Loading MIMIC data for training...');
    const { textData, tabularData, imageData, labels } = await loadAndPreprocessMimicData();
    
    // Prepare training data
    const trainingData = [];
    for (let i = 0; i < Math.min(100, labels.length); i++) {
      const label = labels[i];
      const text = textData.find(t => t.id.includes(i.toString()));
      const tabular = tabularData.find(t => t.patientId === label.patientId);
      const image = imageData.find(img => img.patientId === label.patientId);
      
      trainingData.push({
        text: text?.text,
        tabular: tabular ? {
          age: tabular.age,
          gender: tabular.gender,
          heartRate: tabular.heartRate,
          temperature: tabular.temperature,
          respiratoryRate: tabular.respiratoryRate,
          oxygenSaturation: tabular.oxygenSaturation,
          bloodGlucose: tabular.bloodGlucose
        } : undefined,
        image: image?.data,
        label: label.primaryDiagnosis
      });
    }

    // Train the model
    quantumModel.on('training', (message) => {
      console.log('Training:', message);
    });

    await quantumModel.train(trainingData);
    isModelTrained = true;
    
    console.log('✅ Model training completed successfully');

  } catch (error) {
    console.error('Error during training:', error);
    res.status(500).json({ 
      error: 'Training failed', 
      message: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Predict disease from multimodal input
router.post('/predict', upload.single('image'), async (req: Request, res: Response) => {
  try {
    if (!isModelTrained) {
      return res.status(400).json({ 
        error: 'Model not trained', 
        message: 'Please train the model first using /api/quantum/train' 
      });
    }

    const { textData, tabularData } = req.body;
    let imageData: number[][] | undefined;

    // Process uploaded image if present
    if (req.file) {
      try {
        // Convert image to grayscale matrix
        const { data, info } = await sharp(req.file.buffer)
          .grayscale()
          .resize(64, 64) // Resize to 64x64 for processing
          .raw()
          .toBuffer({ resolveWithObject: true });

        // Convert buffer to 2D array
        imageData = [];
        for (let y = 0; y < info.height; y++) {
          const row = [];
          for (let x = 0; x < info.width; x++) {
            const pixelIndex = y * info.width + x;
            row.push(data[pixelIndex] / 255); // Normalize to 0-1
          }
          imageData.push(row);
        }
      } catch (imageError) {
        console.error('Error processing image:', imageError);
        return res.status(400).json({ 
          error: 'Image processing failed',
          message: 'Could not process the uploaded image'
        });
      }
    }

    // Process tabular data
    let processedTabularData: Record<string, any> | undefined;
    if (tabularData) {
      try {
        processedTabularData = typeof tabularData === 'string' 
          ? JSON.parse(tabularData) 
          : tabularData;
      } catch (parseError) {
        return res.status(400).json({ 
          error: 'Invalid tabular data',
          message: 'Tabular data must be valid JSON'
        });
      }
    }

    console.log('🔄 Making disease prediction with quantum model...');

    // Track prediction progress
    const progressMessages: string[] = [];
    quantumModel.on('processing', (message) => {
      progressMessages.push(message);
      console.log('Processing:', message);
    });

    // Make prediction
    const predictions = await quantumModel.predictDisease(
      textData,
      processedTabularData,
      imageData
    );

    // Get top 5 predictions
    const topPredictions = predictions.slice(0, 5);

    res.json({
      success: true,
      predictions: topPredictions,
      metadata: {
        hasTextData: !!textData,
        hasTabularData: !!processedTabularData,
        hasImageData: !!imageData,
        processingSteps: progressMessages,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).json({ 
      error: 'Prediction failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get model status
router.get('/status', (req: Request, res: Response) => {
  res.json({
    modelTrained: isModelTrained,
    modelType: 'quantum-multimodal-healthcare',
    supportedModalities: ['text', 'tabular', 'image'],
    maxImageSize: '10MB',
    supportedImageFormats: ['JPEG', 'PNG', 'TIFF'],
    availableDiseases: [
      'Pneumonia', 'COVID-19', 'Heart Disease', 'Diabetes', 
      'Hypertension', 'Cancer', 'Stroke', 'COPD', 
      'Kidney Disease', 'Liver Disease'
    ]
  });
});

// Demo endpoint with sample data
router.post('/demo', async (req: Request, res: Response) => {
  try {
    if (!isModelTrained) {
      return res.status(400).json({ 
        error: 'Model not trained', 
        message: 'Please train the model first using /api/quantum/train' 
      });
    }

    // Sample demo data
    const demoText = "Patient presents with chest pain, shortness of breath, and elevated cardiac enzymes. ECG shows ST elevation.";
    const demoTabular = {
      age: 65,
      gender: "Male",
      heartRate: 95,
      bloodPressure: "140/90",
      temperature: 98.6,
      respiratoryRate: 20,
      oxygenSaturation: 94,
      bloodGlucose: 120
    };

    // Generate demo image (simulated ECG)
    const demoImage: number[][] = [];
    for (let y = 0; y < 64; y++) {
      const row = [];
      for (let x = 0; x < 64; x++) {
        // Simulate ECG waveform
        const centerY = 32;
        let waveform = Math.sin(x * 0.3) * 5;
        
        if (x % 15 < 3) {
          waveform += Math.sin(x * 2) * 15; // QRS complex
        }
        
        const waveformY = centerY + waveform;
        const pixel = Math.abs(y - waveformY) < 2 ? 1 : 0.1;
        row.push(pixel);
      }
      demoImage.push(row);
    }

    const predictions = await quantumModel.predictDisease(
      demoText,
      demoTabular,
      demoImage
    );

    res.json({
      success: true,
      demoData: {
        text: demoText,
        tabular: demoTabular,
        imageDescription: "Simulated ECG showing abnormal rhythm"
      },
      predictions: predictions.slice(0, 5),
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error during demo:', error);
    res.status(500).json({ 
      error: 'Demo failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get MIMIC data statistics
router.get('/data-stats', async (req: Request, res: Response) => {
  try {
    const { textData, tabularData, imageData, labels } = await loadAndPreprocessMimicData();
    
    // Calculate statistics
    const textStats = {
      totalRecords: textData.length,
      categories: [...new Set(textData.map(t => t.category))],
      averageTextLength: textData.reduce((sum, t) => sum + t.text.length, 0) / textData.length
    };

    const tabularStats = {
      totalRecords: tabularData.length,
      features: Object.keys(tabularData[0] || {}),
      genderDistribution: tabularData.reduce((acc, t) => {
        acc[t.gender] = (acc[t.gender] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };

    const imageStats = {
      totalRecords: imageData.length,
      imageTypes: [...new Set(imageData.map(img => img.imageType))],
      averageSize: imageData.reduce((sum, img) => sum + img.data.length * img.data[0].length, 0) / imageData.length
    };

    const labelStats = {
      totalRecords: labels.length,
      diagnoses: [...new Set(labels.map(l => l.primaryDiagnosis))],
      severityDistribution: labels.reduce((acc, l) => {
        acc[l.severity] = (acc[l.severity] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
      outcomeDistribution: labels.reduce((acc, l) => {
        acc[l.outcome] = (acc[l.outcome] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };

    res.json({
      success: true,
      statistics: {
        text: textStats,
        tabular: tabularStats,
        image: imageStats,
        labels: labelStats
      },
      dataGeneratedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting data stats:', error);
    res.status(500).json({ 
      error: 'Failed to get data statistics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;