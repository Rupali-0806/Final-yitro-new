import React, { useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Textarea } from '../ui/textarea';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Alert, AlertDescription } from '../ui/alert';
import {
  Upload,
  Brain,
  Activity,
  FileText,
  Database,
  Image as ImageIcon,
  Zap,
  Heart,
  Stethoscope,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Loader2,
} from 'lucide-react';

interface Prediction {
  disease: string;
  probability: number;
}

interface PredictionResponse {
  success: boolean;
  predictions: Prediction[];
  metadata: {
    hasTextData: boolean;
    hasTabularData: boolean;
    hasImageData: boolean;
    processingSteps: string[];
    timestamp: string;
  };
}

export function QuantumHealthcare() {
  const [textData, setTextData] = useState('');
  const [tabularData, setTabularData] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check model status on component mount
  React.useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await fetch('/api/quantum/status');
      const data = await response.json();
      setIsModelTrained(data.modelTrained);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const trainModel = async () => {
    setIsTraining(true);
    setError(null);
    
    try {
      const response = await fetch('/api/quantum/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Training failed');
      }

      // Simulate training progress
      for (let i = 0; i <= 100; i += 10) {
        setProgress(i);
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      setIsModelTrained(true);
      setProgress(100);
      
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImageFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const loadDemo = () => {
    setDemoMode(true);
    setTextData("Patient presents with chest pain, shortness of breath, and elevated cardiac enzymes. ECG shows ST elevation in leads II, III, and aVF. Troponin levels are significantly elevated. Patient has a history of hypertension and smoking.");
    setTabularData(JSON.stringify({
      age: 65,
      gender: "Male",
      heartRate: 95,
      bloodPressure: "140/90",
      temperature: 98.6,
      respiratoryRate: 20,
      oxygenSaturation: 94,
      bloodGlucose: 120,
      cholesterol: 240,
      bmI: 28.5
    }, null, 2));
    setImageFile(null);
    setImagePreview(null);
  };

  const runDemo = async () => {
    setIsLoading(true);
    setError(null);
    setProcessingSteps([]);
    
    try {
      const response = await fetch('/api/quantum/demo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Demo prediction failed');
      }

      const data = await response.json();
      setPredictions(data.predictions);
      setProcessingSteps(['Demo data processed', 'Quantum features extracted', 'Multimodal fusion complete', 'Disease prediction generated']);
      
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Demo failed');
    } finally {
      setIsLoading(false);
    }
  };

  const predictDisease = async () => {
    if (!isModelTrained) {
      setError('Model not trained. Please train the model first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setProcessingSteps([]);
    
    try {
      const formData = new FormData();
      
      if (textData.trim()) {
        formData.append('textData', textData.trim());
      }
      
      if (tabularData.trim()) {
        formData.append('tabularData', tabularData.trim());
      }
      
      if (imageFile) {
        formData.append('image', imageFile);
      }

      const response = await fetch('/api/quantum/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Prediction failed');
      }

      const data: PredictionResponse = await response.json();
      setPredictions(data.predictions);
      setProcessingSteps(data.metadata.processingSteps);
      
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const clearAll = () => {
    setTextData('');
    setTabularData('');
    setImageFile(null);
    setImagePreview(null);
    setPredictions([]);
    setProcessingSteps([]);
    setError(null);
    setDemoMode(false);
  };

  const getProbabilityColor = (probability: number) => {
    if (probability > 0.8) return 'bg-red-500';
    if (probability > 0.6) return 'bg-orange-500';
    if (probability > 0.4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const formatProbability = (probability: number) => {
    return (probability * 100).toFixed(1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <Brain className="h-10 w-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              Quantum Healthcare AI
            </h1>
            <Zap className="h-10 w-10 text-yellow-500" />
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            Multimodal disease prediction using quantum computing and artificial intelligence.
            Upload medical images, enter clinical notes, and provide patient data for accurate diagnosis.
          </p>
        </div>

        {/* Model Status */}
        <Card className="border-2 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Quantum Model Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {isModelTrained ? (
                  <CheckCircle className="h-6 w-6 text-green-500" />
                ) : (
                  <AlertCircle className="h-6 w-6 text-orange-500" />
                )}
                <span className="text-lg font-medium">
                  {isModelTrained ? 'Model Ready' : 'Model Not Trained'}
                </span>
              </div>
              
              {!isModelTrained && (
                <Button 
                  onClick={trainModel} 
                  disabled={isTraining}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {isTraining ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Train Model
                    </>
                  )}
                </Button>
              )}
            </div>
            
            {isTraining && (
              <div className="mt-4">
                <Progress value={progress} className="w-full" />
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Training quantum model with MIMIC-IV data... {progress}%
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Input Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Text Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-blue-600" />
                <span>Clinical Notes</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter clinical notes, symptoms, patient history..."
                value={textData}
                onChange={(e) => setTextData(e.target.value)}
                className="min-h-32"
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Quantum NLP processing for medical text analysis
              </p>
            </CardContent>
          </Card>

          {/* Tabular Data Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Database className="h-5 w-5 text-green-600" />
                <span>Patient Data</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder='{"age": 65, "gender": "Male", "heartRate": 95, "bloodPressure": "140/90", "temperature": 98.6}'
                value={tabularData}
                onChange={(e) => setTabularData(e.target.value)}
                className="min-h-32 font-mono text-sm"
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Enter patient data as JSON for quantum tabular processing
              </p>
            </CardContent>
          </Card>

          {/* Image Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <ImageIcon className="h-5 w-5 text-purple-600" />
                <span>Medical Images</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  variant="outline"
                  className="w-full"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Upload X-ray, CT, MRI
                </Button>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                
                {imagePreview && (
                  <div className="relative">
                    <img
                      src={imagePreview}
                      alt="Medical image preview"
                      className="w-full h-32 object-cover rounded-md border-2 border-gray-200"
                    />
                    <Button
                      onClick={() => {
                        setImageFile(null);
                        setImagePreview(null);
                      }}
                      variant="destructive"
                      size="sm"
                      className="absolute top-2 right-2"
                    >
                      ×
                    </Button>
                  </div>
                )}
                
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Hybrid CNN-Quantum processing for medical imaging
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-4 justify-center">
          <Button
            onClick={loadDemo}
            variant="outline"
            className="border-blue-600 text-blue-600 hover:bg-blue-50"
          >
            <Stethoscope className="h-4 w-4 mr-2" />
            Load Demo Data
          </Button>
          
          {demoMode && (
            <Button
              onClick={runDemo}
              disabled={isLoading}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <TrendingUp className="h-4 w-4 mr-2" />
              )}
              Run Demo Prediction
            </Button>
          )}
          
          <Button
            onClick={predictDisease}
            disabled={isLoading || !isModelTrained}
            className="bg-green-600 hover:bg-green-700"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Heart className="h-4 w-4 mr-2" />
            )}
            Predict Disease
          </Button>
          
          <Button
            onClick={clearAll}
            variant="outline"
            className="border-gray-600 text-gray-600 hover:bg-gray-50"
          >
            Clear All
          </Button>
        </div>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Processing Steps */}
        {processingSteps.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-blue-600" />
                <span>Quantum Processing Pipeline</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {processingSteps.map((step, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm">{step}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Predictions */}
        {predictions.length > 0 && (
          <Card className="border-2 border-green-200 dark:border-green-800">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <span>Quantum Diagnosis Predictions</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {predictions.map((prediction, index) => (
                  <div key={index} className="border rounded-lg p-4 bg-white dark:bg-gray-800">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-semibold">{prediction.disease}</h3>
                      <Badge 
                        variant="secondary"
                        className={`${getProbabilityColor(prediction.probability)} text-white`}
                      >
                        {formatProbability(prediction.probability)}%
                      </Badge>
                    </div>
                    
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${getProbabilityColor(prediction.probability)}`}
                        style={{ width: `${prediction.probability * 100}%` }}
                      />
                    </div>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      Confidence: {prediction.probability > 0.8 ? 'Very High' : 
                                   prediction.probability > 0.6 ? 'High' :
                                   prediction.probability > 0.4 ? 'Medium' : 'Low'}
                    </p>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  <strong>Quantum Processing:</strong> Results generated using multimodal quantum machine learning 
                  with feature fusion from text, tabular, and image data modalities.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}