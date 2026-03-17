import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Item {
    id: irdropPage

    property real minVal: 0
    property real maxVal: 0
    property real meanVal: 0
    property real inferenceTime: 0
    property bool hasResult: false
    property bool inferring: false

    // Track image versions for cache busting
    property int predVersion: 0
    property int colorbarVersion: 0
    property var inputVersions: [0, 0, 0, 0]

    readonly property var channelNames: ["Power Internal", "Power Switching", "Power Scaling", "Power All"]

    Connections {
        target: bridge

        function onIrdropResultReady(min, max, mean, time) {
            minVal = min
            maxVal = max
            meanVal = mean
            inferenceTime = time
            hasResult = true
            inferring = false
        }

        function onInferenceStarted(model) {
            if (model === "irdrop") inferring = true
        }

        function onInferenceFinished(model) {
            if (model === "irdrop") inferring = false
        }

        function onImageUpdated(imageId, version) {
            if (imageId === "irdrop_prediction") {
                predVersion = version
            } else if (imageId === "irdrop_colorbar") {
                colorbarVersion = version
            } else if (imageId.startsWith("irdrop_input_")) {
                var idx = parseInt(imageId.charAt(imageId.length - 1))
                var newVersions = inputVersions.slice()
                newVersions[idx] = version
                inputVersions = newVersions
            }
        }
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 24
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: parent.width
            spacing: 20

            // Header
            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                ColumnLayout {
                    spacing: 4
                    Text {
                        text: "IR Drop Prediction"
                        font.pixelSize: Theme.fontHero
                        font.bold: true
                        color: Theme.textPrimary
                    }
                    Text {
                        text: "Drop a CircuitNet-N14 .npz file to predict voltage drop"
                        font.pixelSize: Theme.fontBody
                        color: Theme.textSecondary
                    }
                }

                Item { Layout.fillWidth: true }

                // Inference indicator
                Rectangle {
                    visible: inferring
                    width: inferRow.width + 24
                    height: 36
                    radius: 18
                    color: Theme.accent + "20"

                    RowLayout {
                        id: inferRow
                        anchors.centerIn: parent
                        spacing: 8

                        BusyIndicator {
                            running: true
                            implicitWidth: 20
                            implicitHeight: 20
                        }
                        Text {
                            text: "Predicting..."
                            font.pixelSize: Theme.fontSmall
                            color: Theme.accent
                        }
                    }
                }
            }

            // Drop zone
            ImageDropZone {
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                title: "Drop .npz Power Map File"
                subtitle: "CircuitNet-N14 format (power_i, power_s, power_sca, power_all)"
                acceptedExtensions: "*.npz"
                filterLabel: "NumPy Archives"

                onFileSelected: function(path) {
                    bridge.predictIRDrop(path)
                }
            }

            // Input channels grid
            Text {
                text: "Input Channels"
                font.pixelSize: Theme.fontTitle
                font.bold: true
                color: Theme.textPrimary
                visible: hasResult
            }

            GridLayout {
                Layout.fillWidth: true
                columns: 4
                rowSpacing: 12
                columnSpacing: 12
                visible: hasResult

                Repeater {
                    model: 4

                    HeatmapViewer {
                        Layout.fillWidth: true
                        Layout.preferredHeight: width
                        imageId: "irdrop_input_" + index
                        imageVersion: inputVersions[index]
                        label: channelNames[index]
                    }
                }
            }

            // Prediction section
            Text {
                text: "Prediction"
                font.pixelSize: Theme.fontTitle
                font.bold: true
                color: Theme.textPrimary
                visible: hasResult
                Layout.topMargin: 8
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 16
                visible: hasResult

                // Prediction heatmap
                HeatmapViewer {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 400
                    imageId: "irdrop_prediction"
                    imageVersion: predVersion
                    label: "Predicted IR Drop"
                    showColorbar: true
                    colorbarId: "irdrop_colorbar"
                    colorbarVersion: colorbarVersion
                    minLabel: minVal.toFixed(4)
                    maxLabel: maxVal.toFixed(4)
                }

                // Statistics panel
                Rectangle {
                    Layout.preferredWidth: 200
                    Layout.preferredHeight: 400
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 16

                        Text {
                            text: "Statistics"
                            font.pixelSize: Theme.fontBody
                            font.bold: true
                            color: Theme.textPrimary
                        }

                        // Min
                        ColumnLayout {
                            spacing: 4

                            Text {
                                text: "Min IR Drop"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                            }

                            Text {
                                text: minVal.toFixed(6)
                                font.pixelSize: Theme.fontTitle
                                font.bold: true
                                color: Theme.success
                            }
                        }

                        // Max
                        ColumnLayout {
                            spacing: 4

                            Text {
                                text: "Max IR Drop"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                            }

                            Text {
                                text: maxVal.toFixed(6)
                                font.pixelSize: Theme.fontTitle
                                font.bold: true
                                color: Theme.error
                            }
                        }

                        // Mean
                        ColumnLayout {
                            spacing: 4

                            Text {
                                text: "Mean IR Drop"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                            }

                            Text {
                                text: meanVal.toFixed(6)
                                font.pixelSize: Theme.fontTitle
                                font.bold: true
                                color: Theme.accent
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.divider
                        }

                        // Inference time
                        ColumnLayout {
                            spacing: 4

                            Text {
                                text: "Inference Time"
                                font.pixelSize: Theme.fontTiny
                                color: Theme.textDim
                            }

                            Text {
                                text: (inferenceTime * 1000).toFixed(1) + " ms"
                                font.pixelSize: Theme.fontTitle
                                font.bold: true
                                color: Theme.warning
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }
            }

            // --- SEVERITY BANNER ---
            Rectangle {
                visible: hasResult && bridge.irdropSeverity !== ""
                Layout.fillWidth: true
                height: sevRow.implicitHeight + 24
                radius: Theme.cardRadius
                color: {
                    var s = bridge.irdropSeverity
                    if (s === "critical") return Theme.error
                    if (s === "high") return Theme.error + "CC"
                    if (s === "medium") return Theme.warning
                    return Theme.success
                }

                RowLayout {
                    id: sevRow
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.margins: 16
                    spacing: 12

                    Rectangle {
                        width: sevLabel.implicitWidth + 16
                        height: sevLabel.implicitHeight + 8
                        radius: 4
                        color: "#00000030"

                        Text {
                            id: sevLabel
                            anchors.centerIn: parent
                            text: bridge.irdropSeverity ? bridge.irdropSeverity.toUpperCase() : ""
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: "#ffffff"
                        }
                    }

                    Text {
                        text: "IR Drop Severity"
                        font.pixelSize: Theme.fontTitle
                        font.bold: true
                        color: bridge.irdropSeverity === "medium" ? Theme.background : "#ffffff"
                        Layout.fillWidth: true
                    }

                    Text {
                        text: bridge.irdropHotspotPct.toFixed(1) + "% hotspot area"
                        font.pixelSize: Theme.fontBody
                        color: bridge.irdropSeverity === "medium" ? Theme.background : "#ffffffCC"
                    }

                    Text {
                        text: (inferenceTime * 1000).toFixed(1) + " ms"
                        font.pixelSize: Theme.fontSmall
                        color: bridge.irdropSeverity === "medium" ? Theme.background + "AA" : "#ffffff99"
                    }
                }
            }

            // --- IR DROP ANALYSIS CARD ---
            Rectangle {
                visible: hasResult && bridge.irdropDescription !== ""
                Layout.fillWidth: true
                implicitHeight: analysisCol.implicitHeight + 32
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                ColumnLayout {
                    id: analysisCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 16
                    spacing: 12

                    Text {
                        text: "IR Drop Analysis"
                        font.pixelSize: Theme.fontTitle
                        font.bold: true
                        color: Theme.textPrimary
                    }

                    Text {
                        text: bridge.irdropDescription
                        font.pixelSize: Theme.fontBody
                        color: Theme.textSecondary
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                        lineHeight: 1.5
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: Theme.divider
                        Layout.topMargin: 4
                        Layout.bottomMargin: 4
                    }

                    Text {
                        text: "Input Channel Guide"
                        font.pixelSize: Theme.fontBody
                        font.bold: true
                        color: Theme.accent
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 24
                        rowSpacing: 8

                        Text {
                            text: "\u25A0 Power Internal (power_i)"
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: Theme.textPrimary
                        }
                        Text {
                            text: "Internal/short-circuit power from transistor switching"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }

                        Text {
                            text: "\u25A0 Power Switching (power_s)"
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: Theme.textPrimary
                        }
                        Text {
                            text: "Dynamic power from charging/discharging load capacitances"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }

                        Text {
                            text: "\u25A0 Power Scaling (power_sca)"
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: Theme.textPrimary
                        }
                        Text {
                            text: "Leakage/static power that scales with technology node"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }

                        Text {
                            text: "\u25A0 Power All (power_all)"
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: Theme.textPrimary
                        }
                        Text {
                            text: "Total combined power consumption across all sources"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textSecondary
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }
                    }
                }
            }

            // --- ROOT CAUSES + RECOMMENDED FIXES ---
            RowLayout {
                visible: hasResult && bridge.irdropCauses.length > 0
                Layout.fillWidth: true
                spacing: 16

                // Left card: Root Causes
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: causesCol.implicitHeight + 32
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        id: causesCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 12

                        Text {
                            text: "Probable Causes"
                            font.pixelSize: Theme.fontTitle
                            font.bold: true
                            color: Theme.error
                        }

                        Repeater {
                            model: bridge.irdropCauses

                            RowLayout {
                                spacing: 10
                                Layout.fillWidth: true

                                Text {
                                    text: "\u2022"
                                    color: Theme.error
                                    font.pixelSize: Theme.fontBody
                                    Layout.alignment: Qt.AlignTop
                                }

                                Text {
                                    text: modelData
                                    font.pixelSize: Theme.fontSmall
                                    color: Theme.textSecondary
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                    lineHeight: 1.4
                                }
                            }
                        }
                    }
                }

                // Right card: Recommended Fixes
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: recsCol.implicitHeight + 32
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        id: recsCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 12

                        Text {
                            text: "Recommended Fixes"
                            font.pixelSize: Theme.fontTitle
                            font.bold: true
                            color: Theme.success
                        }

                        Repeater {
                            model: bridge.irdropRecommendations

                            RowLayout {
                                spacing: 10
                                Layout.fillWidth: true

                                Text {
                                    text: "\u2713"
                                    color: Theme.success
                                    font.pixelSize: Theme.fontBody
                                    font.bold: true
                                    Layout.alignment: Qt.AlignTop
                                }

                                Text {
                                    text: modelData
                                    font.pixelSize: Theme.fontSmall
                                    color: Theme.textSecondary
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                    lineHeight: 1.4
                                }
                            }
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: 32 }
        }
    }
}
