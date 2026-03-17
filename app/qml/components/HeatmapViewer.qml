import QtQuick
import QtQuick.Layouts
import "../theme"

Rectangle {
    id: viewer

    property string imageId: ""
    property int imageVersion: 0
    property string label: ""
    property bool showColorbar: false
    property string colorbarId: ""
    property int colorbarVersion: 0
    property string minLabel: ""
    property string maxLabel: ""

    implicitWidth: 280
    implicitHeight: 280
    radius: Theme.cardRadius
    color: Theme.surface
    border.color: Theme.divider
    border.width: 1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 4

        // Title
        Text {
            text: viewer.label
            font.pixelSize: Theme.fontSmall
            color: Theme.textSecondary
            Layout.alignment: Qt.AlignHCenter
            visible: viewer.label !== ""
        }

        // Image + optional colorbar
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 8

            // Main heatmap image
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 4
                color: Theme.background
                clip: true

                Image {
                    anchors.fill: parent
                    anchors.margins: 2
                    source: viewer.imageId !== "" ? "image://heatmap/" + viewer.imageId + "?v=" + viewer.imageVersion : ""
                    fillMode: Image.PreserveAspectFit
                    cache: false
                    smooth: true
                    visible: viewer.imageId !== ""

                    opacity: 0
                    onStatusChanged: {
                        if (status === Image.Ready) opacity = 1
                    }
                    Behavior on opacity {
                        NumberAnimation { duration: Theme.animNormal }
                    }
                }

                // Placeholder
                Text {
                    anchors.centerIn: parent
                    text: "No Data"
                    font.pixelSize: Theme.fontSmall
                    color: Theme.textDim
                    visible: viewer.imageId === ""
                }
            }

            // Colorbar
            ColumnLayout {
                visible: viewer.showColorbar
                Layout.preferredWidth: 50
                Layout.fillHeight: true
                spacing: 2

                Text {
                    text: viewer.maxLabel
                    font.pixelSize: Theme.fontTiny
                    color: Theme.textDim
                    Layout.alignment: Qt.AlignHCenter
                }

                Rectangle {
                    Layout.fillHeight: true
                    Layout.preferredWidth: 20
                    Layout.alignment: Qt.AlignHCenter
                    radius: 2
                    color: Theme.background
                    clip: true

                    Image {
                        anchors.fill: parent
                        source: viewer.colorbarId !== "" ? "image://heatmap/" + viewer.colorbarId + "?v=" + viewer.colorbarVersion : ""
                        fillMode: Image.Stretch
                        cache: false
                        smooth: true
                    }
                }

                Text {
                    text: viewer.minLabel
                    font.pixelSize: Theme.fontTiny
                    color: Theme.textDim
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }
    }
}
