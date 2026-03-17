import QtQuick
import QtQuick.Layouts
import "../theme"

Rectangle {
    id: statusBar

    property string deviceName: "N/A"
    property bool cudaAvailable: false
    property string gpuMemory: "N/A"
    property bool defectLoaded: false
    property bool modelsLoading: false

    height: 32
    color: Theme.surface

    // Top border
    Rectangle {
        width: parent.width
        height: 1
        color: Theme.divider
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        spacing: 20

        // GPU Status
        RowLayout {
            spacing: 6

            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: statusBar.cudaAvailable ? Theme.success : Theme.warning

                SequentialAnimation on opacity {
                    running: statusBar.modelsLoading
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.3; duration: 500 }
                    NumberAnimation { to: 1.0; duration: 500 }
                }
            }

            Text {
                text: statusBar.cudaAvailable ? "GPU: " + statusBar.deviceName : "CPU Mode"
                font.pixelSize: Theme.fontTiny
                color: Theme.textDim
            }

            Text {
                text: "(" + statusBar.gpuMemory + ")"
                font.pixelSize: Theme.fontTiny
                color: Theme.textDim
                visible: statusBar.cudaAvailable
            }
        }

        // Separator
        Rectangle {
            width: 1
            height: 16
            color: Theme.divider
        }

        // Model statuses
        RowLayout {
            spacing: 6

            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: statusBar.defectLoaded ? Theme.success : Theme.textDim
            }

            Text {
                text: "Defect"
                font.pixelSize: Theme.fontTiny
                color: statusBar.defectLoaded ? Theme.textSecondary : Theme.textDim
            }
        }

        Item { Layout.fillWidth: true }

        // Loading indicator
        Text {
            text: statusBar.modelsLoading ? "Loading models..." : "Ready"
            font.pixelSize: Theme.fontTiny
            color: statusBar.modelsLoading ? Theme.warning : Theme.textDim

            SequentialAnimation on opacity {
                running: statusBar.modelsLoading
                loops: Animation.Infinite
                NumberAnimation { to: 0.4; duration: 600 }
                NumberAnimation { to: 1.0; duration: 600 }
            }
        }
    }
}
