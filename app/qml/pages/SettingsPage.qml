import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Item {
    id: settingsPage

    ScrollView {
        anchors.fill: parent
        anchors.margins: 32
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: parent.width
            spacing: 24

            // Hero
            ColumnLayout {
                spacing: 4
                Text {
                    text: "Settings"
                    font.pixelSize: Theme.fontHero
                    font.bold: true
                    color: Theme.textPrimary
                }
                Text {
                    text: "Configuration and system information"
                    font.pixelSize: Theme.fontBody
                    color: Theme.textSecondary
                }
            }

            // Device Info
            Rectangle {
                Layout.fillWidth: true
                height: deviceCol.implicitHeight + 40
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                ColumnLayout {
                    id: deviceCol
                    anchors.fill: parent
                    anchors.margins: Theme.cardPadding
                    spacing: 16

                    Text {
                        text: "Compute Device"
                        font.pixelSize: Theme.fontTitle
                        font.bold: true
                        color: Theme.textPrimary
                    }

                    GridLayout {
                        columns: 2
                        rowSpacing: 12
                        columnSpacing: 24
                        Layout.fillWidth: true

                        Text { text: "Device"; font.pixelSize: Theme.fontSmall; color: Theme.textDim }
                        Text { text: bridge.deviceName; font.pixelSize: Theme.fontSmall; color: Theme.textPrimary; font.bold: true }

                        Text { text: "CUDA"; font.pixelSize: Theme.fontSmall; color: Theme.textDim }
                        RowLayout {
                            spacing: 8
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: bridge.cudaAvailable ? Theme.success : Theme.error
                            }
                            Text {
                                text: bridge.cudaAvailable ? "Available" : "Not Available"
                                font.pixelSize: Theme.fontSmall
                                color: bridge.cudaAvailable ? Theme.success : Theme.error
                            }
                        }

                        Text { text: "GPU Memory"; font.pixelSize: Theme.fontSmall; color: Theme.textDim }
                        Text { text: bridge.gpuMemory; font.pixelSize: Theme.fontSmall; color: Theme.textPrimary }

                        Text { text: "PyTorch"; font.pixelSize: Theme.fontSmall; color: Theme.textDim }
                        Text { text: bridge.torchVersion; font.pixelSize: Theme.fontSmall; color: Theme.textPrimary }
                    }
                }
            }

            // Models Section
            Rectangle {
                Layout.fillWidth: true
                height: modelCol.implicitHeight + 40
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                ColumnLayout {
                    id: modelCol
                    anchors.fill: parent
                    anchors.margins: Theme.cardPadding
                    spacing: 16

                    Text {
                        text: "Models"
                        font.pixelSize: Theme.fontTitle
                        font.bold: true
                        color: Theme.textPrimary
                    }

                    // Defect Model
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Rectangle {
                            width: 10; height: 10; radius: 5
                            color: bridge.defectLoaded ? Theme.success : Theme.textDim
                        }

                        ColumnLayout {
                            spacing: 2
                            Layout.fillWidth: true

                            Text {
                                text: "Defect Classification Model"
                                font.pixelSize: Theme.fontBody
                                color: Theme.textPrimary
                            }
                            Text {
                                text: "problem_a/checkpoints/best_model.pt"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                                font.family: "Consolas"
                            }
                        }

                        Rectangle {
                            width: defStatus.width + 16
                            height: 22
                            radius: 11
                            color: bridge.defectLoaded ? Theme.success + "30" : Theme.textDim + "30"

                            Text {
                                id: defStatus
                                anchors.centerIn: parent
                                text: bridge.defectLoaded ? "LOADED" : "NOT LOADED"
                                font.pixelSize: Theme.fontTiny
                                font.bold: true
                                color: bridge.defectLoaded ? Theme.success : Theme.textDim
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: Theme.divider
                    }

                    // IR Drop Model
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Rectangle {
                            width: 10; height: 10; radius: 5
                            color: bridge.irdropLoaded ? Theme.success : Theme.textDim
                        }

                        ColumnLayout {
                            spacing: 2
                            Layout.fillWidth: true

                            Text {
                                text: "IR Drop Prediction Model"
                                font.pixelSize: Theme.fontBody
                                color: Theme.textPrimary
                            }
                            Text {
                                text: "problem_d/checkpoints/best_model.pt"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                                font.family: "Consolas"
                            }
                        }

                        Rectangle {
                            width: irStatus.width + 16
                            height: 22
                            radius: 11
                            color: bridge.irdropLoaded ? Theme.success + "30" : Theme.textDim + "30"

                            Text {
                                id: irStatus
                                anchors.centerIn: parent
                                text: bridge.irdropLoaded ? "LOADED" : "NOT LOADED"
                                font.pixelSize: Theme.fontTiny
                                font.bold: true
                                color: bridge.irdropLoaded ? Theme.success : Theme.textDim
                            }
                        }
                    }

                    // Reload button
                    Rectangle {
                        Layout.preferredWidth: 160
                        Layout.preferredHeight: 40
                        Layout.topMargin: 8
                        radius: 8
                        color: reloadHover.containsMouse ? Theme.accent : Theme.accent + "CC"

                        Text {
                            anchors.centerIn: parent
                            text: bridge.modelsLoading ? "Loading..." : "Reload Models"
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: Theme.background
                        }

                        MouseArea {
                            id: reloadHover
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            enabled: !bridge.modelsLoading
                            onClicked: bridge.loadModels()
                        }

                        Behavior on color {
                            ColorAnimation { duration: Theme.animFast }
                        }
                    }
                }
            }

            // About
            Rectangle {
                Layout.fillWidth: true
                height: aboutCol.implicitHeight + 40
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                ColumnLayout {
                    id: aboutCol
                    anchors.fill: parent
                    anchors.margins: Theme.cardPadding
                    spacing: 12

                    Text {
                        text: "About"
                        font.pixelSize: Theme.fontTitle
                        font.bold: true
                        color: Theme.textPrimary
                    }

                    Text {
                        text: "SemiAI Evaluation Suite v1.0.0"
                        font.pixelSize: Theme.fontBody
                        color: Theme.accent
                        font.bold: true
                    }

                    Text {
                        text: "A professional evaluation tool for semiconductor ML models.\n" +
                              "Built for the Semiconductor Solutions Challenge 2026.\n\n" +
                              "Problem A: Few-Shot Defect Classification\n" +
                              "Problem D: Static Voltage Drop Prediction\n\n" +
                              "Built with PySide6 + QML + PyTorch"
                        font.pixelSize: Theme.fontSmall
                        color: Theme.textSecondary
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                        lineHeight: 1.4
                    }
                }
            }

            Item { Layout.preferredHeight: 32 }
        }
    }
}
