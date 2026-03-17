import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "theme"
import "components"
import "pages"

ApplicationWindow {
    id: window
    visible: true
    width: 1400
    height: 900
    minimumWidth: 1100
    minimumHeight: 700
    title: "SemiAI - Semiconductor Evaluation Suite"
    color: Theme.background

    // Error dialog
    Dialog {
        id: errorDialog
        title: "Error"
        modal: true
        anchors.centerIn: parent
        width: 420
        standardButtons: Dialog.Ok

        property string errorTitle: ""
        property string errorMessage: ""

        background: Rectangle {
            color: Theme.surface
            radius: Theme.cardRadius
            border.color: Theme.error + "60"
            border.width: 1
        }

        header: Rectangle {
            color: "transparent"
            height: 48

            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8

                Text {
                    text: "\u26A0"
                    font.pixelSize: 18
                    color: Theme.error
                }

                Text {
                    text: errorDialog.errorTitle
                    font.pixelSize: Theme.fontTitle
                    font.bold: true
                    color: Theme.textPrimary
                }
            }
        }

        contentItem: Text {
            text: errorDialog.errorMessage
            font.pixelSize: Theme.fontBody
            color: Theme.textSecondary
            wrapMode: Text.Wrap
            padding: 16
        }
    }

    Connections {
        target: bridge

        function onErrorOccurred(title, message) {
            errorDialog.errorTitle = title
            errorDialog.errorMessage = message
            errorDialog.open()
        }
    }

    // Main layout
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Sidebar
        Sidebar {
            id: sidebar
            Layout.fillHeight: true

            onPageSelected: function(index) {
                pageStack.currentIndex = index
            }
        }

        // Content area
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // Page stack
            StackLayout {
                id: pageStack
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: 0

                DashboardPage {}
                DefectPage {}
                SettingsPage {}
            }

            // Status bar
            StatusBar {
                Layout.fillWidth: true
                deviceName: bridge.deviceName
                cudaAvailable: bridge.cudaAvailable
                gpuMemory: bridge.gpuMemory
                defectLoaded: bridge.defectLoaded
                modelsLoading: bridge.modelsLoading
            }
        }
    }

}
