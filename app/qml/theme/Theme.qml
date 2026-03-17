pragma Singleton
import QtQuick

QtObject {
    // Colors
    readonly property color background:     "#0a0e14"
    readonly property color surface:        "#1a1e2e"
    readonly property color surfaceLight:   "#242840"
    readonly property color accent:         "#00bcd4"
    readonly property color accentDark:     "#009688"
    readonly property color success:        "#4caf50"
    readonly property color warning:        "#ffc107"
    readonly property color error:          "#f44336"
    readonly property color textPrimary:    "#ffffff"
    readonly property color textSecondary:  "#b0bec5"
    readonly property color textDim:        "#607d8b"
    readonly property color divider:        "#2a2e3e"

    // Sidebar
    readonly property int sidebarWidth:     240

    // Card
    readonly property int cardRadius:       12
    readonly property int cardPadding:      20

    // Animation durations (ms)
    readonly property int animFast:         150
    readonly property int animNormal:       250
    readonly property int animSlow:         400

    // Fonts
    readonly property int fontTiny:         10
    readonly property int fontSmall:        12
    readonly property int fontBody:         14
    readonly property int fontTitle:        18
    readonly property int fontHeader:       24
    readonly property int fontHero:         32
}
