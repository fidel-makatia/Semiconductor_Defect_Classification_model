import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Item {
    id: dashboardPage

    Flickable {
        anchors.fill: parent
        anchors.margins: 32
        contentWidth: width
        contentHeight: mainCol.implicitHeight
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ColumnLayout {
            id: mainCol
            width: parent.width
            spacing: 24

            // Hero
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4
                Text {
                    text: "Dashboard"
                    font.pixelSize: Theme.fontHero
                    font.bold: true
                    color: Theme.textPrimary
                }
                Text {
                    text: "System overview and model status"
                    font.pixelSize: Theme.fontBody
                    color: Theme.textSecondary
                }
            }

            // Metric cards row
            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                MetricCard {
                    title: "Compute Device"
                    value: bridge.deviceName
                    subtitle: bridge.cudaAvailable ? "CUDA Accelerated" : "CPU Only"
                    iconText: "\u2699"
                    accentColor: bridge.cudaAvailable ? Theme.success : Theme.warning
                    Layout.preferredWidth: 260
                    Layout.preferredHeight: 130
                }

                MetricCard {
                    title: "GPU Memory"
                    value: bridge.gpuMemory
                    subtitle: "Total VRAM"
                    iconText: "\u2630"
                    accentColor: Theme.warning
                    Layout.preferredWidth: 260
                    Layout.preferredHeight: 130
                    visible: bridge.cudaAvailable
                }

                MetricCard {
                    title: "PyTorch"
                    value: bridge.torchVersion
                    subtitle: bridge.cudaAvailable ? "CUDA Enabled" : "CPU Build"
                    iconText: "\u26A1"
                    accentColor: Theme.accent
                    Layout.preferredWidth: 260
                    Layout.preferredHeight: 130
                }

                Item { Layout.fillWidth: true }
            }

            // Model Status Section
            Text {
                text: "Model Status"
                font.pixelSize: Theme.fontTitle
                font.bold: true
                color: Theme.textPrimary
                Layout.topMargin: 8
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                // Defect Model Card
                Rectangle {
                    Layout.preferredWidth: 380
                    Layout.preferredHeight: 190
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: Theme.cardPadding
                        spacing: 10

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Rectangle {
                                width: 10; height: 10; radius: 5
                                color: bridge.defectLoaded ? Theme.success : Theme.textDim

                                SequentialAnimation on opacity {
                                    running: bridge.modelsLoading && !bridge.defectLoaded
                                    loops: Animation.Infinite
                                    NumberAnimation { to: 0.3; duration: 500 }
                                    NumberAnimation { to: 1.0; duration: 500 }
                                }
                            }

                            Text {
                                text: "Defect Classification"
                                font.pixelSize: Theme.fontTitle
                                font.bold: true
                                color: Theme.textPrimary
                                Layout.fillWidth: true
                            }

                            Rectangle {
                                width: statusLabel1.width + 16
                                height: 22; radius: 11
                                color: bridge.defectLoaded ? Theme.success + "30" : Theme.textDim + "30"
                                Text {
                                    id: statusLabel1
                                    anchors.centerIn: parent
                                    text: bridge.defectLoaded ? "LOADED" : "PENDING"
                                    font.pixelSize: Theme.fontTiny; font.bold: true
                                    color: bridge.defectLoaded ? Theme.success : Theme.textDim
                                }
                            }
                        }

                        Text {
                            text: "DINOv2 ViT-L/14 + Prototypical Network"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            Layout.fillWidth: true
                        }

                        RowLayout {
                            spacing: 32
                            ColumnLayout {
                                spacing: 2
                                Text { text: "Epoch"; font.pixelSize: Theme.fontTiny; color: Theme.textDim }
                                Text { text: bridge.defectEpoch; font.pixelSize: Theme.fontBody; font.bold: true; color: Theme.textPrimary }
                            }
                            ColumnLayout {
                                spacing: 2
                                Text { text: "Best Accuracy"; font.pixelSize: Theme.fontTiny; color: Theme.textDim }
                                Text { text: bridge.defectAccuracy; font.pixelSize: Theme.fontBody; font.bold: true; color: Theme.accent }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }

                Item { Layout.fillWidth: true }
            }

            // Architecture info
            Text {
                text: "Architecture Details"
                font.pixelSize: Theme.fontTitle
                font.bold: true
                color: Theme.textPrimary
                Layout.topMargin: 8
            }

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: archCol.implicitHeight + 40
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                RowLayout {
                    id: archCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: Theme.cardPadding
                    spacing: 40

                    // Problem A
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 6

                        Text {
                            text: "Problem A: Few-Shot Defect Classification"
                            font.pixelSize: Theme.fontBody
                            font.bold: true
                            color: Theme.accent
                            Layout.fillWidth: true
                        }

                        Text {
                            text: "\u2022 Backbone: DINOv2 ViT-L/14 (304M frozen params)\n\u2022 Head: Prototypical Network (trainable projection)\n\u2022 Input: Grayscale up to 7000x5600 (Intel defect images)\n\u2022 Method: Incremental prototype tracking\n\u2022 Few-shot: K=1 to K=20 support images\n\u2022 Cosine similarity + learned temperature"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            lineHeight: 1.5
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: 32 }
        }
    }
}
