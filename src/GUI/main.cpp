#include "MainForm.h"

using namespace System;
using namespace System::Windows::Forms;

[STAThreadAttribute]
int main(cli::array<String^>^ args)
{
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    
    YoloTrainerGUI::MainForm form;
    Application::Run(%form);
    
    return 0;
}
