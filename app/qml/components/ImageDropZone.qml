import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../theme"

Rectangle {
    id: dropZone

    property string title: "Drop Image Here"
    property string subtitle: "or click to browse"
    property string acceptedExtensions: "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
    property string imageSource: ""
    property bool hasImage: imageSource !== ""
    property string filterLabel: "Images"

    signal fileDropped(string filePath)
    signal fileSelected(string filePath)

    implicitWidth: 300
    implicitHeight: 250
    radius: Theme.cardRadius
    color: Theme.surface
    border.color: dropArea.containsDrag ? Theme.accent : (hoverArea.containsMouse ? Theme.surfaceLight : Theme.divider)
    border.width: dropArea.containsDrag ? 2 : 1

    Behavior on border.color {
        ColorAnimation { duration: Theme.animFast }
    }

    // Dashed border effect when no image
    Rectangle {
        anchors.fill: parent
        anchors.margins: 8
        radius: Theme.cardRadius - 4
        color: "transparent"
        border.color: Theme.divider
        border.width: hasImage ? 0 : 1
        opacity: 0.5
        visible: !hasImage
    }

    // Image display
    Image {
        id: displayImage
        anchors.fill: parent
        anchors.margins: 4
        source: dropZone.imageSource
        fillMode: Image.PreserveAspectFit
        visible: hasImage
        cache: false
        smooth: true

        // Fade in animation
        opacity: 0
        Component.onCompleted: opacity = 1
        Behavior on opacity {
            NumberAnimation { duration: Theme.animNormal }
        }
    }

    // Placeholder content
    ColumnLayout {
        anchors.centerIn: parent
        spacing: 12
        visible: !hasImage

        Text {
            text: "\u2B07"
            font.pixelSize: 36
            color: dropArea.containsDrag ? Theme.accent : Theme.textDim
            Layout.alignment: Qt.AlignHCenter
            opacity: dropArea.containsDrag ? 1.0 : 0.6

            Behavior on color {
                ColorAnimation { duration: Theme.animFast }
            }

            // Gentle bounce animation when dragging
            SequentialAnimation on y {
                running: dropArea.containsDrag
                loops: Animation.Infinite
                NumberAnimation { to: -5; duration: 400; easing.type: Easing.InOutQuad }
                NumberAnimation { to: 0; duration: 400; easing.type: Easing.InOutQuad }
            }
        }

        Text {
            text: dropZone.title
            font.pixelSize: Theme.fontBody
            color: Theme.textSecondary
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            text: dropZone.subtitle
            font.pixelSize: Theme.fontSmall
            color: Theme.textDim
            Layout.alignment: Qt.AlignHCenter
        }
    }

    // Drop area overlay
    DropArea {
        id: dropArea
        anchors.fill: parent

        onDropped: function(drop) {
            if (drop.hasUrls && drop.urls.length > 0) {
                var url = drop.urls[0].toString()
                // Strip file:/// prefix
                if (Qt.platform.os === "windows") {
                    url = url.replace("file:///", "")
                } else {
                    url = url.replace("file://", "")
                }
                url = decodeURIComponent(url)
                dropZone.fileDropped(url)
                dropZone.fileSelected(url)
            }
        }
    }

    // Click to browse
    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: fileDialog.open()
    }

    FileDialog {
        id: fileDialog
        title: "Select File"
        nameFilters: [filterLabel + " (" + acceptedExtensions + ")", "All files (*)"]
        onAccepted: {
            var url = selectedFile.toString()
            if (Qt.platform.os === "windows") {
                url = url.replace("file:///", "")
            } else {
                url = url.replace("file://", "")
            }
            url = decodeURIComponent(url)
            dropZone.fileDropped(url)
            dropZone.fileSelected(url)
        }
    }

    // Drag highlight overlay
    Rectangle {
        anchors.fill: parent
        radius: parent.radius
        color: Theme.accent
        opacity: dropArea.containsDrag ? 0.08 : 0
        visible: opacity > 0

        Behavior on opacity {
            NumberAnimation { duration: Theme.animFast }
        }
    }
}
