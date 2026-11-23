cmake_minimum_required(VERSION 3.21)
project(search_demo LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick QuickControls2 Multimedia)
qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(search_demo main.cpp)
qt_add_qml_module(search_demo
    URI SearchDemo
    VERSION 1.0
    QML_FILES Main.qml ChatPanel.qml)

target_link_libraries(search_demo PRIVATE Qt6::Quick Qt6::QuickControls2 Qt6::Multimedia)
set_target_properties(search_demo PROPERTIES WIN32_EXECUTABLE TRUE)
