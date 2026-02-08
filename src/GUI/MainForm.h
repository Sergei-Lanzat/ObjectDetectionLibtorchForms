#pragma once

#include "../include/YoloModel.h"
#include "../include/YoloTrainer.h"
#include <msclr/marshal_cppstd.h>
#include <memory>
#include <vector>

namespace YoloTrainerGUI {

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;
    using namespace System::Windows::Forms::DataVisualization::Charting;

    /// <summary>
    /// Main training form for YOLO object detection
    /// </summary>
    public ref class MainForm : public System::Windows::Forms::Form
    {
    public:
        MainForm(void)
        {
            InitializeComponent();
            InitializeChart();
            device_ = new torch::Device(torch::kCPU);
        }

    protected:
        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        ~MainForm()
        {
            if (components)
            {
                delete components;
            }
            if (device_)
            {
                delete device_;
            }
        }

    private:
        // UI Controls
        System::Windows::Forms::GroupBox^ gbDataset;
        System::Windows::Forms::Label^ lblDatasetPath;
        System::Windows::Forms::TextBox^ txtDatasetPath;
        System::Windows::Forms::Button^ btnBrowseDataset;
        System::Windows::Forms::Label^ lblValidationPath;
        System::Windows::Forms::TextBox^ txtValidationPath;
        System::Windows::Forms::Button^ btnBrowseValidation;
        System::Windows::Forms::Button^ btnConvertAnnotations;

        System::Windows::Forms::GroupBox^ gbModelConfig;
        System::Windows::Forms::Label^ lblModelType;
        System::Windows::Forms::ComboBox^ cmbModelType;
        System::Windows::Forms::Label^ lblNumClasses;
        System::Windows::Forms::NumericUpDown^ numClasses;
        System::Windows::Forms::Label^ lblInputSize;
        System::Windows::Forms::NumericUpDown^ numInputSize;
        System::Windows::Forms::CheckBox^ chkLoadPretrained;
        System::Windows::Forms::TextBox^ txtPretrainedPath;
        System::Windows::Forms::Button^ btnBrowsePretrained;

        System::Windows::Forms::GroupBox^ gbTrainingConfig;
        System::Windows::Forms::Label^ lblEpochs;
        System::Windows::Forms::NumericUpDown^ numEpochs;
        System::Windows::Forms::Label^ lblBatchSize;
        System::Windows::Forms::NumericUpDown^ numBatchSize;
        System::Windows::Forms::Label^ lblLearningRate;
        System::Windows::Forms::TextBox^ txtLearningRate;
        System::Windows::Forms::Label^ lblOptimizer;
        System::Windows::Forms::ComboBox^ cmbOptimizer;
        System::Windows::Forms::CheckBox^ chkFreezeBackbone;
        System::Windows::Forms::NumericUpDown^ numFreezeEpochs;
        System::Windows::Forms::CheckBox^ chkUseGPU;

        System::Windows::Forms::GroupBox^ gbTrainingControl;
        System::Windows::Forms::Button^ btnStartTraining;
        System::Windows::Forms::Button^ btnStopTraining;
        System::Windows::Forms::Button^ btnExportOnnx;
        System::Windows::Forms::ProgressBar^ progressBar;
        System::Windows::Forms::Label^ lblStatus;

        System::Windows::Forms::GroupBox^ gbVisualization;
        System::Windows::Forms::DataVisualization::Charting::Chart^ chartLoss;
        System::Windows::Forms::TextBox^ txtLog;

        System::ComponentModel::Container^ components;

        // Native objects
        torch::Device* device_;
        std::shared_ptr<yolo::YoloModel>* model_;
        std::shared_ptr<yolo::YoloTrainer>* trainer_;
        System::Threading::Thread^ trainingThread_;

        bool isTraining_;
        System::Collections::Generic::List<double>^ lossHistory_;
        System::Collections::Generic::List<int>^ epochHistory_;

#pragma region Windows Form Designer generated code
        void InitializeComponent(void)
        {
            this->components = gcnew System::ComponentModel::Container();
            
            // Form settings
            this->Text = L"YOLO Object Detection Trainer";
            this->Size = System::Drawing::Size(1400, 900);
            this->StartPosition = FormStartPosition::CenterScreen;

            // Initialize collections
            lossHistory_ = gcnew System::Collections::Generic::List<double>();
            epochHistory_ = gcnew System::Collections::Generic::List<int>();
            isTraining_ = false;

            // Dataset Group
            this->gbDataset = gcnew GroupBox();
            this->gbDataset->Text = L"Dataset Configuration";
            this->gbDataset->Location = Point(10, 10);
            this->gbDataset->Size = System::Drawing::Size(450, 150);

            this->lblDatasetPath = gcnew Label();
            this->lblDatasetPath->Text = L"Training Dataset Path:";
            this->lblDatasetPath->Location = Point(10, 25);
            this->lblDatasetPath->AutoSize = true;

            this->txtDatasetPath = gcnew TextBox();
            this->txtDatasetPath->Location = Point(10, 45);
            this->txtDatasetPath->Size = System::Drawing::Size(340, 20);

            this->btnBrowseDataset = gcnew Button();
            this->btnBrowseDataset->Text = L"Browse...";
            this->btnBrowseDataset->Location = Point(355, 43);
            this->btnBrowseDataset->Size = System::Drawing::Size(80, 23);
            this->btnBrowseDataset->Click += gcnew EventHandler(this, &MainForm::btnBrowseDataset_Click);

            this->lblValidationPath = gcnew Label();
            this->lblValidationPath->Text = L"Validation Dataset Path:";
            this->lblValidationPath->Location = Point(10, 75);
            this->lblValidationPath->AutoSize = true;

            this->txtValidationPath = gcnew TextBox();
            this->txtValidationPath->Location = Point(10, 95);
            this->txtValidationPath->Size = System::Drawing::Size(340, 20);

            this->btnBrowseValidation = gcnew Button();
            this->btnBrowseValidation->Text = L"Browse...";
            this->btnBrowseValidation->Location = Point(355, 93);
            this->btnBrowseValidation->Size = System::Drawing::Size(80, 23);
            this->btnBrowseValidation->Click += gcnew EventHandler(this, &MainForm::btnBrowseValidation_Click);

            this->btnConvertAnnotations = gcnew Button();
            this->btnConvertAnnotations->Text = L"Convert Annotations (COCO/VOC to YOLO)";
            this->btnConvertAnnotations->Location = Point(10, 120);
            this->btnConvertAnnotations->Size = System::Drawing::Size(250, 23);
            this->btnConvertAnnotations->Click += gcnew EventHandler(this, &MainForm::btnConvertAnnotations_Click);

            this->gbDataset->Controls->Add(this->lblDatasetPath);
            this->gbDataset->Controls->Add(this->txtDatasetPath);
            this->gbDataset->Controls->Add(this->btnBrowseDataset);
            this->gbDataset->Controls->Add(this->lblValidationPath);
            this->gbDataset->Controls->Add(this->txtValidationPath);
            this->gbDataset->Controls->Add(this->btnBrowseValidation);
            this->gbDataset->Controls->Add(this->btnConvertAnnotations);

            // Model Configuration Group
            this->gbModelConfig = gcnew GroupBox();
            this->gbModelConfig->Text = L"Model Configuration";
            this->gbModelConfig->Location = Point(470, 10);
            this->gbModelConfig->Size = System::Drawing::Size(450, 150);

            this->lblModelType = gcnew Label();
            this->lblModelType->Text = L"Model Type:";
            this->lblModelType->Location = Point(10, 25);
            this->lblModelType->AutoSize = true;

            this->cmbModelType = gcnew ComboBox();
            this->cmbModelType->Location = Point(120, 22);
            this->cmbModelType->Size = System::Drawing::Size(150, 21);
            this->cmbModelType->DropDownStyle = ComboBoxStyle::DropDownList;
            this->cmbModelType->Items->AddRange(gcnew cli::array<Object^> {
                L"YOLOv5s", L"YOLOv5m", L"YOLOv5l", L"YOLOv8s", L"YOLOv8m", L"YOLO26L"
            });
            this->cmbModelType->SelectedIndex = 0;

            this->lblNumClasses = gcnew Label();
            this->lblNumClasses->Text = L"Number of Classes:";
            this->lblNumClasses->Location = Point(10, 55);
            this->lblNumClasses->AutoSize = true;

            this->numClasses = gcnew NumericUpDown();
            this->numClasses->Location = Point(120, 53);
            this->numClasses->Size = System::Drawing::Size(80, 20);
            this->numClasses->Minimum = 1;
            this->numClasses->Maximum = 1000;
            this->numClasses->Value = 80;

            this->lblInputSize = gcnew Label();
            this->lblInputSize->Text = L"Input Size:";
            this->lblInputSize->Location = Point(10, 85);
            this->lblInputSize->AutoSize = true;

            this->numInputSize = gcnew NumericUpDown();
            this->numInputSize->Location = Point(120, 83);
            this->numInputSize->Size = System::Drawing::Size(80, 20);
            this->numInputSize->Minimum = 320;
            this->numInputSize->Maximum = 1280;
            this->numInputSize->Increment = 32;
            this->numInputSize->Value = 640;

            this->chkLoadPretrained = gcnew CheckBox();
            this->chkLoadPretrained->Text = L"Load Pre-trained Weights";
            this->chkLoadPretrained->Location = Point(10, 115);
            this->chkLoadPretrained->AutoSize = true;
            this->chkLoadPretrained->CheckedChanged += gcnew EventHandler(this, &MainForm::chkLoadPretrained_CheckedChanged);

            this->txtPretrainedPath = gcnew TextBox();
            this->txtPretrainedPath->Location = Point(180, 113);
            this->txtPretrainedPath->Size = System::Drawing::Size(180, 20);
            this->txtPretrainedPath->Enabled = false;

            this->btnBrowsePretrained = gcnew Button();
            this->btnBrowsePretrained->Text = L"...";
            this->btnBrowsePretrained->Location = Point(365, 111);
            this->btnBrowsePretrained->Size = System::Drawing::Size(40, 23);
            this->btnBrowsePretrained->Enabled = false;
            this->btnBrowsePretrained->Click += gcnew EventHandler(this, &MainForm::btnBrowsePretrained_Click);

            this->gbModelConfig->Controls->Add(this->lblModelType);
            this->gbModelConfig->Controls->Add(this->cmbModelType);
            this->gbModelConfig->Controls->Add(this->lblNumClasses);
            this->gbModelConfig->Controls->Add(this->numClasses);
            this->gbModelConfig->Controls->Add(this->lblInputSize);
            this->gbModelConfig->Controls->Add(this->numInputSize);
            this->gbModelConfig->Controls->Add(this->chkLoadPretrained);
            this->gbModelConfig->Controls->Add(this->txtPretrainedPath);
            this->gbModelConfig->Controls->Add(this->btnBrowsePretrained);

            // Training Configuration Group
            this->gbTrainingConfig = gcnew GroupBox();
            this->gbTrainingConfig->Text = L"Training Configuration";
            this->gbTrainingConfig->Location = Point(930, 10);
            this->gbTrainingConfig->Size = System::Drawing::Size(450, 150);

            this->lblEpochs = gcnew Label();
            this->lblEpochs->Text = L"Epochs:";
            this->lblEpochs->Location = Point(10, 25);
            this->lblEpochs->AutoSize = true;

            this->numEpochs = gcnew NumericUpDown();
            this->numEpochs->Location = Point(150, 23);
            this->numEpochs->Size = System::Drawing::Size(80, 20);
            this->numEpochs->Minimum = 1;
            this->numEpochs->Maximum = 1000;
            this->numEpochs->Value = 100;

            this->lblBatchSize = gcnew Label();
            this->lblBatchSize->Text = L"Batch Size:";
            this->lblBatchSize->Location = Point(10, 55);
            this->lblBatchSize->AutoSize = true;

            this->numBatchSize = gcnew NumericUpDown();
            this->numBatchSize->Location = Point(150, 53);
            this->numBatchSize->Size = System::Drawing::Size(80, 20);
            this->numBatchSize->Minimum = 1;
            this->numBatchSize->Maximum = 128;
            this->numBatchSize->Value = 16;

            this->lblLearningRate = gcnew Label();
            this->lblLearningRate->Text = L"Learning Rate:";
            this->lblLearningRate->Location = Point(10, 85);
            this->lblLearningRate->AutoSize = true;

            this->txtLearningRate = gcnew TextBox();
            this->txtLearningRate->Location = Point(150, 83);
            this->txtLearningRate->Size = System::Drawing::Size(80, 20);
            this->txtLearningRate->Text = L"0.01";

            this->lblOptimizer = gcnew Label();
            this->lblOptimizer->Text = L"Optimizer:";
            this->lblOptimizer->Location = Point(250, 25);
            this->lblOptimizer->AutoSize = true;

            this->cmbOptimizer = gcnew ComboBox();
            this->cmbOptimizer->Location = Point(330, 23);
            this->cmbOptimizer->Size = System::Drawing::Size(100, 21);
            this->cmbOptimizer->DropDownStyle = ComboBoxStyle::DropDownList;
            this->cmbOptimizer->Items->AddRange(gcnew cli::array<Object^> {
                L"SGD", L"Adam", L"AdamW"
            });
            this->cmbOptimizer->SelectedIndex = 0;

            this->chkFreezeBackbone = gcnew CheckBox();
            this->chkFreezeBackbone->Text = L"Freeze Backbone for";
            this->chkFreezeBackbone->Location = Point(250, 55);
            this->chkFreezeBackbone->AutoSize = true;

            this->numFreezeEpochs = gcnew NumericUpDown();
            this->numFreezeEpochs->Location = Point(370, 53);
            this->numFreezeEpochs->Size = System::Drawing::Size(50, 20);
            this->numFreezeEpochs->Minimum = 0;
            this->numFreezeEpochs->Maximum = 100;
            this->numFreezeEpochs->Value = 5;

            this->chkUseGPU = gcnew CheckBox();
            this->chkUseGPU->Text = L"Use GPU (if available)";
            this->chkUseGPU->Location = Point(250, 85);
            this->chkUseGPU->AutoSize = true;
            this->chkUseGPU->Checked = false;

            this->gbTrainingConfig->Controls->Add(this->lblEpochs);
            this->gbTrainingConfig->Controls->Add(this->numEpochs);
            this->gbTrainingConfig->Controls->Add(this->lblBatchSize);
            this->gbTrainingConfig->Controls->Add(this->numBatchSize);
            this->gbTrainingConfig->Controls->Add(this->lblLearningRate);
            this->gbTrainingConfig->Controls->Add(this->txtLearningRate);
            this->gbTrainingConfig->Controls->Add(this->lblOptimizer);
            this->gbTrainingConfig->Controls->Add(this->cmbOptimizer);
            this->gbTrainingConfig->Controls->Add(this->chkFreezeBackbone);
            this->gbTrainingConfig->Controls->Add(this->numFreezeEpochs);
            this->gbTrainingConfig->Controls->Add(this->chkUseGPU);

            // Training Control Group
            this->gbTrainingControl = gcnew GroupBox();
            this->gbTrainingControl->Text = L"Training Control";
            this->gbTrainingControl->Location = Point(10, 170);
            this->gbTrainingControl->Size = System::Drawing::Size(450, 120);

            this->btnStartTraining = gcnew Button();
            this->btnStartTraining->Text = L"Start Training";
            this->btnStartTraining->Location = Point(10, 25);
            this->btnStartTraining->Size = System::Drawing::Size(200, 40);
            this->btnStartTraining->Font = gcnew System::Drawing::Font(L"Arial", 12, FontStyle::Bold);
            this->btnStartTraining->BackColor = Color::LightGreen;
            this->btnStartTraining->Click += gcnew EventHandler(this, &MainForm::btnStartTraining_Click);

            this->btnStopTraining = gcnew Button();
            this->btnStopTraining->Text = L"Stop Training";
            this->btnStopTraining->Location = Point(220, 25);
            this->btnStopTraining->Size = System::Drawing::Size(200, 40);
            this->btnStopTraining->Font = gcnew System::Drawing::Font(L"Arial", 12, FontStyle::Bold);
            this->btnStopTraining->BackColor = Color::LightCoral;
            this->btnStopTraining->Enabled = false;
            this->btnStopTraining->Click += gcnew EventHandler(this, &MainForm::btnStopTraining_Click);

            this->btnExportOnnx = gcnew Button();
            this->btnExportOnnx->Text = L"Export to ONNX";
            this->btnExportOnnx->Location = Point(10, 75);
            this->btnExportOnnx->Size = System::Drawing::Size(200, 30);
            this->btnExportOnnx->Click += gcnew EventHandler(this, &MainForm::btnExportOnnx_Click);

            this->progressBar = gcnew ProgressBar();
            this->progressBar->Location = Point(220, 75);
            this->progressBar->Size = System::Drawing::Size(200, 30);

            this->lblStatus = gcnew Label();
            this->lblStatus->Text = L"Status: Ready";
            this->lblStatus->Location = Point(10, 108);
            this->lblStatus->AutoSize = true;

            this->gbTrainingControl->Controls->Add(this->btnStartTraining);
            this->gbTrainingControl->Controls->Add(this->btnStopTraining);
            this->gbTrainingControl->Controls->Add(this->btnExportOnnx);
            this->gbTrainingControl->Controls->Add(this->progressBar);
            this->gbTrainingControl->Controls->Add(this->lblStatus);

            // Visualization Group
            this->gbVisualization = gcnew GroupBox();
            this->gbVisualization->Text = L"Training Visualization";
            this->gbVisualization->Location = Point(10, 300);
            this->gbVisualization->Size = System::Drawing::Size(1370, 540);

            this->chartLoss = gcnew Chart();
            this->chartLoss->Location = Point(10, 20);
            this->chartLoss->Size = System::Drawing::Size(850, 500);

            this->txtLog = gcnew TextBox();
            this->txtLog->Location = Point(870, 20);
            this->txtLog->Size = System::Drawing::Size(490, 500);
            this->txtLog->Multiline = true;
            this->txtLog->ScrollBars = ScrollBars::Vertical;
            this->txtLog->ReadOnly = true;
            this->txtLog->Font = gcnew System::Drawing::Font(L"Consolas", 9);

            this->gbVisualization->Controls->Add(this->chartLoss);
            this->gbVisualization->Controls->Add(this->txtLog);

            // Add all groups to form
            this->Controls->Add(this->gbDataset);
            this->Controls->Add(this->gbModelConfig);
            this->Controls->Add(this->gbTrainingConfig);
            this->Controls->Add(this->gbTrainingControl);
            this->Controls->Add(this->gbVisualization);
        }
#pragma endregion

        void InitializeChart();
        void UpdateChart();
        void AppendLog(String^ message);
        
        // Event handlers
        void btnBrowseDataset_Click(Object^ sender, EventArgs^ e);
        void btnBrowseValidation_Click(Object^ sender, EventArgs^ e);
        void btnConvertAnnotations_Click(Object^ sender, EventArgs^ e);
        void btnBrowsePretrained_Click(Object^ sender, EventArgs^ e);
        void chkLoadPretrained_CheckedChanged(Object^ sender, EventArgs^ e);
        void btnStartTraining_Click(Object^ sender, EventArgs^ e);
        void btnStopTraining_Click(Object^ sender, EventArgs^ e);
        void btnExportOnnx_Click(Object^ sender, EventArgs^ e);
        
        void TrainingThreadProc();
        void OnTrainingProgress(float loss, int epoch, int iteration);
    };
}
