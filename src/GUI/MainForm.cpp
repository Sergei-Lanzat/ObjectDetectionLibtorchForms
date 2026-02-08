#include "MainForm.h"
#include <msclr/marshal_cppstd.h>

namespace YoloTrainerGUI {

void MainForm::InitializeChart() {
    // Configure chart
    chartLoss->ChartAreas->Add(gcnew ChartArea("MainArea"));
    chartLoss->Series->Add(gcnew Series("Training Loss"));
    
    chartLoss->Series["Training Loss"]->ChartType = SeriesChartType::Line;
    chartLoss->Series["Training Loss"]->BorderWidth = 2;
    chartLoss->Series["Training Loss"]->Color = Color::Blue;
    
    chartLoss->ChartAreas["MainArea"]->AxisX->Title = "Epoch";
    chartLoss->ChartAreas["MainArea"]->AxisY->Title = "Loss";
    chartLoss->ChartAreas["MainArea"]->AxisX->MajorGrid->LineColor = Color::LightGray;
    chartLoss->ChartAreas["MainArea"]->AxisY->MajorGrid->LineColor = Color::LightGray;
    
    chartLoss->Titles->Add(gcnew Title("Training Loss Convergence"));
    chartLoss->Titles[0]->Font = gcnew System::Drawing::Font("Arial", 14, FontStyle::Bold);
}

void MainForm::UpdateChart() {
    if (this->InvokeRequired) {
        this->Invoke(gcnew Action(this, &MainForm::UpdateChart));
        return;
    }
    
    chartLoss->Series["Training Loss"]->Points->Clear();
    
    for (int i = 0; i < epochHistory_->Count && i < lossHistory_->Count; i++) {
        chartLoss->Series["Training Loss"]->Points->AddXY(
            epochHistory_[i], 
            lossHistory_[i]
        );
    }
    
    chartLoss->Invalidate();
}

void MainForm::AppendLog(String^ message) {
    if (this->InvokeRequired) {
        this->Invoke(gcnew Action<String^>(this, &MainForm::AppendLog), message);
        return;
    }
    
    String^ timestamp = DateTime::Now.ToString("HH:mm:ss");
    txtLog->AppendText(String::Format("[{0}] {1}\r\n", timestamp, message));
}

void MainForm::btnBrowseDataset_Click(Object^ sender, EventArgs^ e) {
    FolderBrowserDialog^ dialog = gcnew FolderBrowserDialog();
    dialog->Description = "Select training dataset folder";
    
    if (dialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
        txtDatasetPath->Text = dialog->SelectedPath;
        AppendLog("Training dataset path set: " + dialog->SelectedPath);
    }
}

void MainForm::btnBrowseValidation_Click(Object^ sender, EventArgs^ e) {
    FolderBrowserDialog^ dialog = gcnew FolderBrowserDialog();
    dialog->Description = "Select validation dataset folder";
    
    if (dialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
        txtValidationPath->Text = dialog->SelectedPath;
        AppendLog("Validation dataset path set: " + dialog->SelectedPath);
    }
}

void MainForm::btnConvertAnnotations_Click(Object^ sender, EventArgs^ e) {
    // Show dialog to select annotation format and convert
    Form^ convertForm = gcnew Form();
    convertForm->Text = "Convert Annotations";
    convertForm->Size = System::Drawing::Size(500, 300);
    convertForm->StartPosition = FormStartPosition::CenterParent;
    
    Label^ lblFormat = gcnew Label();
    lblFormat->Text = "Annotation Format:";
    lblFormat->Location = Point(20, 20);
    lblFormat->AutoSize = true;
    convertForm->Controls->Add(lblFormat);
    
    ComboBox^ cmbFormat = gcnew ComboBox();
    cmbFormat->Location = Point(20, 45);
    cmbFormat->Size = System::Drawing::Size(200, 21);
    cmbFormat->DropDownStyle = ComboBoxStyle::DropDownList;
    cmbFormat->Items->AddRange(gcnew cli::array<Object^> {
        "COCO JSON", "Pascal VOC XML", "LabelImg XML"
    });
    cmbFormat->SelectedIndex = 0;
    convertForm->Controls->Add(cmbFormat);
    
    Label^ lblInput = gcnew Label();
    lblInput->Text = "Input Path:";
    lblInput->Location = Point(20, 80);
    lblInput->AutoSize = true;
    convertForm->Controls->Add(lblInput);
    
    TextBox^ txtInput = gcnew TextBox();
    txtInput->Location = Point(20, 105);
    txtInput->Size = System::Drawing::Size(350, 20);
    convertForm->Controls->Add(txtInput);
    
    Button^ btnBrowseInput = gcnew Button();
    btnBrowseInput->Text = "Browse...";
    btnBrowseInput->Location = Point(380, 103);
    btnBrowseInput->Click += gcnew EventHandler([txtInput](Object^ s, EventArgs^ e) {
        OpenFileDialog^ dlg = gcnew OpenFileDialog();
        if (dlg->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
            txtInput->Text = dlg->FileName;
        }
    });
    convertForm->Controls->Add(btnBrowseInput);
    
    Label^ lblOutput = gcnew Label();
    lblOutput->Text = "Output Directory:";
    lblOutput->Location = Point(20, 140);
    lblOutput->AutoSize = true;
    convertForm->Controls->Add(lblOutput);
    
    TextBox^ txtOutput = gcnew TextBox();
    txtOutput->Location = Point(20, 165);
    txtOutput->Size = System::Drawing::Size(350, 20);
    convertForm->Controls->Add(txtOutput);
    
    Button^ btnBrowseOutput = gcnew Button();
    btnBrowseOutput->Text = "Browse...";
    btnBrowseOutput->Location = Point(380, 163);
    btnBrowseOutput->Click += gcnew EventHandler([txtOutput](Object^ s, EventArgs^ e) {
        FolderBrowserDialog^ dlg = gcnew FolderBrowserDialog();
        if (dlg->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
            txtOutput->Text = dlg->SelectedPath;
        }
    });
    convertForm->Controls->Add(btnBrowseOutput);
    
    Button^ btnConvert = gcnew Button();
    btnConvert->Text = "Convert";
    btnConvert->Location = Point(20, 210);
    btnConvert->Size = System::Drawing::Size(100, 30);
    btnConvert->Click += gcnew EventHandler([this, cmbFormat, txtInput, txtOutput, convertForm](Object^ s, EventArgs^ e) {
        String^ inputPath = txtInput->Text;
        String^ outputPath = txtOutput->Text;
        
        if (String::IsNullOrEmpty(inputPath) || String::IsNullOrEmpty(outputPath)) {
            MessageBox::Show("Please specify input and output paths", "Error", 
                           MessageBoxButtons::OK, MessageBoxIcon::Error);
            return;
        }
        
        msclr::interop::marshal_context context;
        std::string input_std = context.marshal_as<std::string>(inputPath);
        std::string output_std = context.marshal_as<std::string>(outputPath);
        
        bool success = false;
        switch (cmbFormat->SelectedIndex) {
            case 0: // COCO
                success = yolo::AnnotationConverter::coco_to_yolo(input_std, output_std);
                break;
            case 1: // Pascal VOC
                success = yolo::AnnotationConverter::voc_to_yolo(input_std, output_std);
                break;
            case 2: // LabelImg
                success = yolo::AnnotationConverter::labelimg_to_yolo(input_std, output_std);
                break;
        }
        
        if (success) {
            MessageBox::Show("Annotations converted successfully!", "Success",
                           MessageBoxButtons::OK, MessageBoxIcon::Information);
            AppendLog("Annotations converted to: " + outputPath);
            convertForm->Close();
        } else {
            MessageBox::Show("Failed to convert annotations", "Error",
                           MessageBoxButtons::OK, MessageBoxIcon::Error);
        }
    });
    convertForm->Controls->Add(btnConvert);
    
    convertForm->ShowDialog();
}

void MainForm::btnBrowsePretrained_Click(Object^ sender, EventArgs^ e) {
    OpenFileDialog^ dialog = gcnew OpenFileDialog();
    dialog->Filter = "PyTorch Models (*.pt)|*.pt|All Files (*.*)|*.*";
    dialog->Title = "Select Pre-trained Model";
    
    if (dialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
        txtPretrainedPath->Text = dialog->FileName;
        AppendLog("Pre-trained model path set: " + dialog->FileName);
    }
}

void MainForm::chkLoadPretrained_CheckedChanged(Object^ sender, EventArgs^ e) {
    txtPretrainedPath->Enabled = chkLoadPretrained->Checked;
    btnBrowsePretrained->Enabled = chkLoadPretrained->Checked;
}

void MainForm::btnStartTraining_Click(Object^ sender, EventArgs^ e) {
    // Validate inputs
    if (String::IsNullOrEmpty(txtDatasetPath->Text)) {
        MessageBox::Show("Please specify training dataset path", "Error",
                       MessageBoxButtons::OK, MessageBoxIcon::Error);
        return;
    }
    
    // Clear history
    lossHistory_->Clear();
    epochHistory_->Clear();
    txtLog->Clear();
    
    AppendLog("=== Starting YOLO Training ===");
    AppendLog("Model Type: " + cmbModelType->Text);
    AppendLog("Number of Classes: " + numClasses->Value.ToString());
    AppendLog("Input Size: " + numInputSize->Value.ToString());
    AppendLog("Epochs: " + numEpochs->Value.ToString());
    AppendLog("Batch Size: " + numBatchSize->Value.ToString());
    AppendLog("Learning Rate: " + txtLearningRate->Text);
    
    // Update UI
    btnStartTraining->Enabled = false;
    btnStopTraining->Enabled = true;
    lblStatus->Text = "Status: Training...";
    progressBar->Value = 0;
    isTraining_ = true;
    
    // Start training in separate thread
    trainingThread_ = gcnew System::Threading::Thread(
        gcnew System::Threading::ThreadStart(this, &MainForm::TrainingThreadProc)
    );
    trainingThread_->Start();
}

void MainForm::btnStopTraining_Click(Object^ sender, EventArgs^ e) {
    AppendLog("Stopping training...");
    isTraining_ = false;
    
    // Stop the trainer if it exists
    if (trainer_ != nullptr && *trainer_ != nullptr) {
        (*trainer_)->stop_training();
    }
    
    btnStartTraining->Enabled = true;
    btnStopTraining->Enabled = false;
    lblStatus->Text = "Status: Stopped";
}

void MainForm::btnExportOnnx_Click(Object^ sender, EventArgs^ e) {
    if (model_ == nullptr || *model_ == nullptr) {
        MessageBox::Show("No trained model available. Please train a model first.", "Error",
                       MessageBoxButtons::OK, MessageBoxIcon::Error);
        return;
    }
    
    SaveFileDialog^ dialog = gcnew SaveFileDialog();
    dialog->Filter = "ONNX Models (*.onnx)|*.onnx|All Files (*.*)|*.*";
    dialog->Title = "Export Model to ONNX";
    dialog->DefaultExt = "onnx";
    
    if (dialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
        msclr::interop::marshal_context context;
        std::string onnx_path = context.marshal_as<std::string>(dialog->FileName);
        
        AppendLog("Exporting model to ONNX: " + dialog->FileName);
        
        bool success = (*model_)->export_onnx(onnx_path);
        
        if (success) {
            MessageBox::Show("Model exported successfully!", "Success",
                           MessageBoxButtons::OK, MessageBoxIcon::Information);
            AppendLog("Model exported successfully!");
        } else {
            MessageBox::Show("Failed to export model", "Error",
                           MessageBoxButtons::OK, MessageBoxIcon::Error);
            AppendLog("Failed to export model");
        }
    }
}

void MainForm::TrainingThreadProc() {
    try {
        // Setup device
        if (chkUseGPU->Checked && torch::cuda::is_available()) {
            delete device_;
            device_ = new torch::Device(torch::kCUDA);
            AppendLog("Using GPU for training");
        } else {
            delete device_;
            device_ = new torch::Device(torch::kCPU);
            AppendLog("Using CPU for training");
        }
        
        // Create model configuration
        yolo::YoloConfig config;
        config.num_classes = (int)numClasses->Value;
        config.input_width = (int)numInputSize->Value;
        config.input_height = (int)numInputSize->Value;
        
        msclr::interop::marshal_context context;
        config.model_type = context.marshal_as<std::string>(cmbModelType->Text->ToLower());
        
        // Create model
        AppendLog("Creating YOLO model...");
        model_ = new std::shared_ptr<yolo::YoloModel>(
            std::make_shared<yolo::YoloModel>(config)
        );
        (*model_)->to(*device_);
        
        // Load pre-trained weights if specified
        if (chkLoadPretrained->Checked && !String::IsNullOrEmpty(txtPretrainedPath->Text)) {
            std::string pretrained_path = context.marshal_as<std::string>(txtPretrainedPath->Text);
            AppendLog("Loading pre-trained weights...");
            (*model_)->load_pretrained(pretrained_path);
        }
        
        // Create training configuration
        yolo::TrainingConfig train_config;
        train_config.num_epochs = (int)numEpochs->Value;
        train_config.batch_size = (int)numBatchSize->Value;
        train_config.learning_rate = float::Parse(txtLearningRate->Text);
        train_config.optimizer_type = context.marshal_as<std::string>(cmbOptimizer->Text);
        train_config.freeze_backbone = chkFreezeBackbone->Checked;
        train_config.freeze_epochs = (int)numFreezeEpochs->Value;
        
        // Create trainer
        AppendLog("Initializing trainer...");
        trainer_ = new std::shared_ptr<yolo::YoloTrainer>(
            std::make_shared<yolo::YoloTrainer>(*model_, train_config, *device_)
        );
        
        // Prepare dataset paths
        std::string train_path = context.marshal_as<std::string>(txtDatasetPath->Text);
        std::string val_path = String::IsNullOrEmpty(txtValidationPath->Text) ? 
                               "" : context.marshal_as<std::string>(txtValidationPath->Text);
        
        // Define progress callback
        auto progress_callback = [this](const yolo::TrainingMetrics& metrics) {
            this->OnTrainingProgress(metrics.loss, metrics.epoch, metrics.iteration);
        };
        
        // Start training
        AppendLog("Starting training loop...");
        bool success = (*trainer_)->train(train_path, val_path, progress_callback);
        
        if (success) {
            AppendLog("=== Training Completed Successfully ===");
            MessageBox::Show("Training completed successfully!", "Success",
                           MessageBoxButtons::OK, MessageBoxIcon::Information);
        } else {
            AppendLog("=== Training Failed ===");
            MessageBox::Show("Training failed. Check logs for details.", "Error",
                           MessageBoxButtons::OK, MessageBoxIcon::Error);
        }
        
    } catch (System::Exception^ ex) {
        AppendLog("Error: " + ex->Message);
        MessageBox::Show("Training error: " + ex->Message, "Error",
                       MessageBoxButtons::OK, MessageBoxIcon::Error);
    } catch (const std::exception& ex) {
        String^ errorMsg = gcnew String(ex.what());
        AppendLog("Error: " + errorMsg);
        MessageBox::Show("Training error: " + errorMsg, "Error",
                       MessageBoxButtons::OK, MessageBoxIcon::Error);
    }
    
    // Reset UI
    this->Invoke(gcnew Action([this]() {
        btnStartTraining->Enabled = true;
        btnStopTraining->Enabled = false;
        lblStatus->Text = "Status: Completed";
        isTraining_ = false;
    }));
}

void MainForm::OnTrainingProgress(float loss, int epoch, int iteration) {
    // Update loss history
    if (this->InvokeRequired) {
        this->Invoke(gcnew Action<float, int, int>(this, &MainForm::OnTrainingProgress), 
                    loss, epoch, iteration);
        return;
    }
    
    // Add to history
    if (epochHistory_->Count == 0 || epochHistory_[epochHistory_->Count - 1] != epoch) {
        epochHistory_->Add(epoch);
        lossHistory_->Add((double)loss);
    } else {
        lossHistory_[lossHistory_->Count - 1] = (double)loss;
    }
    
    // Update chart
    UpdateChart();
    
    // Update progress bar
    int progress = (int)((float)epoch / numEpochs->Value * 100);
    progressBar->Value = Math::Min(progress, 100);
    
    // Update status
    lblStatus->Text = String::Format("Epoch: {0}/{1}, Loss: {2:F4}", 
                                     epoch + 1, (int)numEpochs->Value, loss);
}

} // namespace YoloTrainerGUI
