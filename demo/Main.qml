import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtMultimedia

// 이벤트 기반 의미 검색 데모 — 다보이조 검색 화면과 같은 3단(조건 | 결과 | 상세·재생)
// 서버: C++ search_server (WSL) — /search, /img, /info
ApplicationWindow {
    id: win
    width: 1500; height: 900; visible: true
    title: "이벤트 기반 의미 검색 데모"
    color: c.bg

    QtObject {
        id: c
        readonly property color bg: "#12151b"
        readonly property color panel: "#1a1e26"
        readonly property color card: "#222733"
        readonly property color border: "#2e3442"
        readonly property color text: "#e6e9ef"
        readonly property color sub: "#8b93a3"
        readonly property color accent: "#f37321"     // 한화 오렌지 계열
        readonly property color event: "#3fb68b"
        readonly property color frame: "#7a8aa8"
        readonly property color dup: "#d9534f"
    }

    property string server: typeof serverUrl !== "undefined" ? serverUrl : "http://localhost:8765"
    property string mode: "compare"            // event | frame | compare
    property var evRes: null
    property var frRes: null
    property var sel: null                     // 선택한 결과 1건
    property string selKind: ""
    property bool busy: false
    property var info: ({})

    function get(path, cb) {
        const x = new XMLHttpRequest()
        x.onreadystatechange = function () {
            if (x.readyState !== XMLHttpRequest.DONE) return
            if (x.status === 200) cb(JSON.parse(x.responseText))
            else { busy = false; status.text = "서버 응답 없음 (" + server + ")" }
        }
        x.open("GET", server + path)
        x.send()
    }
    function mmss(sec) {
        const s = Math.max(0, Math.floor(sec))
        return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0")
    }
    function search(text) {
        if (text !== undefined) query.text = text
        const q = query.text.trim()
        if (q === "") return
        busy = true; sel = null; player.stop()
        evRes = null; frRes = null
        const enc = encodeURIComponent(q)
        let pending = mode === "compare" ? 2 : 1
        const done = function () { if (--pending === 0) busy = false }
        if (mode !== "frame") get("/search?index=event&k=12&q=" + enc, function (r) { evRes = r; done() })
        if (mode !== "event") get("/search?index=frame&k=12&q=" + enc, function (r) { frRes = r; done() })
    }
    function pick(h, kind) {
        sel = h; selKind = kind
        player.source = h.video
        player.position = Math.max(0, h.t * 1000 - 1000)
        player.play()
    }

    Component.onCompleted: get("/info", function (r) { info = r; status.text = "연결됨 · " + server })

    MediaPlayer { id: player; videoOutput: vout; audioOutput: AudioOutput { muted: true }
        onMediaStatusChanged: if (mediaStatus === MediaPlayer.LoadedMedia && sel) position = Math.max(0, sel.t * 1000 - 1000) }

    // ── 결과 칸 하나 (이벤트 또는 프레임) ──
    component ResultColumn: ColumnLayout {
        id: col
        property var res: null
        property string kind: "event"
        property color tint: kind === "event" ? c.event : c.frame
        spacing: 8

        Rectangle {
            Layout.fillWidth: true; implicitHeight: hdr.implicitHeight + 18
            color: c.card; radius: 4; border.color: c.border
            Rectangle { width: 3; height: parent.height; color: col.tint; radius: 2 }
            Column {
                id: hdr; x: 14; anchors.verticalCenter: parent.verticalCenter; spacing: 3
                Text { text: col.kind === "event" ? "이벤트 방식 — 사람이 지나갈 때 1번" : "프레임 방식 — 초당 1번, 화면 속 사람 전부"
                       color: c.text; font.pixelSize: 14; font.bold: true }
                Text {
                    color: c.sub; font.pixelSize: 12
                    text: {
                        const n = win.info[col.kind] ? win.info[col.kind].embedded.toLocaleString(Qt.locale(), "f", 0) : "-"
                        if (!col.res) return "임베딩 " + n + "개"
                        return "임베딩 " + n + "개 · 검색 " + col.res.ms.toFixed(1) + " ms"
                    }
                }
                Text {
                    visible: col.res !== null
                    // 추적 ID 기준: 같은 트랙을 여러 번 저장한 것만 '반복'으로 셈 (끊긴 다른 트랙의 같은 사람은 못 셈 → 보수적)
                    text: col.res ? "상위 " + col.res.hits.length + "개 중 같은 트랙 반복 " + (col.res.hits.length - col.res.unique) + "개" : ""
                    color: col.res && col.res.unique < col.res.hits.length ? c.dup : col.tint
                    font.pixelSize: 13; font.bold: true
                }
            }
        }

        GridView {
            id: grid
            Layout.fillWidth: true; Layout.fillHeight: true
            clip: true
            cellWidth: Math.floor(width / Math.max(2, Math.floor(width / 150)))
            cellHeight: cellWidth * 1.9
            model: col.res ? col.res.hits : []
            ScrollBar.vertical: ScrollBar {}
            delegate: Item {
                required property var modelData
                required property int index
                width: grid.cellWidth; height: grid.cellHeight
                // 앞 순위에 이미 나온 트랙이면 '중복' (같은 트랙을 여러 번 저장한 프레임 방식의 비용)
                property bool dup: {
                    const hs = col.res.hits
                    for (let i = 0; i < index; ++i) if (hs[i].person === modelData.person) return true
                    return false
                }
                Rectangle {
                    anchors.fill: parent; anchors.margins: 4
                    color: c.card; radius: 4
                    border.width: win.sel === modelData ? 2 : 1
                    border.color: win.sel === modelData ? c.accent : (parent.dup ? c.dup : c.border)
                    opacity: parent.dup ? 0.55 : 1.0
                    Image {
                        id: img
                        anchors { left: parent.left; right: parent.right; top: parent.top; margins: 4 }
                        height: parent.height - meta.height - 12
                        source: win.server + modelData.img
                        fillMode: Image.PreserveAspectFit; asynchronous: true; cache: true
                    }
                    Rectangle {
                        x: 8; y: 8; width: rk.implicitWidth + 10; height: 18; radius: 3; color: "#000000aa"
                        Text { id: rk; anchors.centerIn: parent; text: "#" + modelData.rank; color: "white"; font.pixelSize: 11; font.bold: true }
                    }
                    Rectangle {
                        visible: parent.parent.dup
                        anchors { right: parent.right; top: parent.top; margins: 8 }
                        width: 34; height: 18; radius: 3; color: c.dup
                        Text { anchors.centerIn: parent; text: "중복"; color: "white"; font.pixelSize: 11; font.bold: true }
                    }
                    Column {
                        id: meta
                        anchors { left: parent.left; right: parent.right; bottom: parent.bottom; margins: 6 }
                        spacing: 1
                        Text { text: modelData.score.toFixed(3) + "  ·  " + modelData.channel; color: c.text; font.pixelSize: 11 }
                        Text { text: "영상 " + win.mmss(modelData.t) + (modelData.members.length > 1 ? "  ·  조각 " + modelData.members.length + "개 이음" : "")
                               color: c.sub; font.pixelSize: 11 }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: win.pick(modelData, col.kind) }
                }
            }
        }
    }

    RowLayout {
        anchors.fill: parent; anchors.margins: 14; spacing: 12

        // ── 좌: 검색 조건 ──
        Rectangle {
            Layout.preferredWidth: 270; Layout.fillHeight: true
            color: c.panel; radius: 4; border.color: c.border
            ColumnLayout {
                anchors.fill: parent; anchors.margins: 14; spacing: 10
                Text { text: "의미 검색"; color: c.text; font.pixelSize: 18; font.bold: true }
                Text { text: "카메라 메타데이터(퇴장 시 베스트샷)로 만든 인덱스를 자연어로 검색"; color: c.sub
                       font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Item { height: 4 }
                Text { text: "질문 (영어)"; color: c.sub; font.pixelSize: 11; font.bold: true }
                TextField {
                    id: query; Layout.fillWidth: true
                    placeholderText: "a person in a red jacket"
                    color: c.text; font.pixelSize: 13
                    background: Rectangle { color: c.bg; radius: 3; border.color: query.activeFocus ? c.accent : c.border }
                    onAccepted: win.search()
                }
                Text { text: "보기"; color: c.sub; font.pixelSize: 11; font.bold: true }
                RowLayout {
                    Layout.fillWidth: true; spacing: 4
                    Repeater {
                        model: [["compare", "비교"], ["event", "이벤트"], ["frame", "프레임"]]
                        Rectangle {
                            required property var modelData
                            Layout.fillWidth: true; height: 30; radius: 3
                            color: win.mode === modelData[0] ? c.accent : c.bg
                            border.color: win.mode === modelData[0] ? c.accent : c.border
                            Text { anchors.centerIn: parent; text: modelData[1]; font.pixelSize: 12; font.bold: true
                                   color: win.mode === modelData[0] ? "white" : c.sub }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                        onClicked: { win.mode = modelData[0]; if (query.text !== "") win.search() } }
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true; height: 36; radius: 3
                    color: win.busy ? c.border : c.accent
                    Text { anchors.centerIn: parent; text: win.busy ? "검색 중…" : "검색"; color: "white"; font.pixelSize: 14; font.bold: true }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: if (!win.busy) win.search() }
                }
                Item { height: 6 }
                Text { text: "예시 질문"; color: c.sub; font.pixelSize: 11; font.bold: true }
                Repeater {
                    model: ["a woman in a red top sitting", "a person in a neon green hoodie",
                            "a person carrying white plastic bags", "an elderly woman with white hair sitting",
                            "a person wearing a neon yellow beanie", "a person loading bags into a car"]
                    Rectangle {
                        required property string modelData
                        Layout.fillWidth: true; height: chipT.implicitHeight + 14; radius: 3
                        color: c.bg; border.color: chipMa.containsMouse ? c.accent : c.border
                        Text { id: chipT; x: 9; width: parent.width - 18; anchors.verticalCenter: parent.verticalCenter
                               text: modelData; wrapMode: Text.Wrap; font.pixelSize: 12
                               color: chipMa.containsMouse ? c.text : c.sub }
                        MouseArea { id: chipMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: win.search(modelData) }
                    }
                }
                Item { Layout.fillHeight: true }
                Text { id: status; text: "서버 연결 중…"; color: c.sub; font.pixelSize: 11; wrapMode: Text.Wrap; Layout.fillWidth: true }
            }
        }

        // ── 중: 결과 ──
        RowLayout {
            Layout.fillWidth: true; Layout.fillHeight: true; spacing: 10
            ResultColumn { visible: win.mode !== "frame"; Layout.fillWidth: true; Layout.fillHeight: true; kind: "event"; res: win.evRes }
            ResultColumn { visible: win.mode !== "event"; Layout.fillWidth: true; Layout.fillHeight: true; kind: "frame"; res: win.frRes }
        }

        // ── 우: 상세·재생 ──
        Rectangle {
            Layout.preferredWidth: 380; Layout.fillHeight: true
            color: c.panel; radius: 4; border.color: c.border
            ColumnLayout {
                anchors.fill: parent; anchors.margins: 14; spacing: 10
                Text { text: "선택한 결과"; color: c.text; font.pixelSize: 15; font.bold: true }
                Text { visible: !win.sel; text: "가운데에서 결과를 고르면 베스트샷과 그 시각 영상을 보여줍니다."
                       color: c.sub; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true }
                RowLayout {
                    visible: win.sel !== null; Layout.fillWidth: true; spacing: 10
                    Image { Layout.preferredWidth: 120; Layout.preferredHeight: 220
                            source: win.sel ? win.server + win.sel.img : ""; fillMode: Image.PreserveAspectFit }
                    Column {
                        Layout.fillWidth: true; spacing: 5
                        Text { text: win.sel ? win.sel.channel : ""; color: c.text; font.pixelSize: 14; font.bold: true }
                        Text { text: win.sel ? (win.selKind === "event" ? "이벤트 인덱스" : "프레임 인덱스") : ""
                               color: win.selKind === "event" ? c.event : c.frame; font.pixelSize: 12; font.bold: true }
                        Text { text: win.sel ? "유사도 " + win.sel.score.toFixed(3) : ""; color: c.sub; font.pixelSize: 12 }
                        Text { text: win.sel ? "영상 " + win.mmss(win.sel.t) + (win.sel.t_end > win.sel.t ? " ~ " + win.mmss(win.sel.t_end) : "") : ""
                               color: c.sub; font.pixelSize: 12 }
                        Text { visible: win.sel && win.sel.members.length > 1
                               text: win.sel ? "끊긴 트랙 조각 " + win.sel.members.length + "개를 한 사람으로 이음" : ""
                               color: c.event; font.pixelSize: 12; wrapMode: Text.Wrap; width: parent.width }
                        Text { visible: win.sel && win.sel.top !== ""
                               text: win.sel ? "옷 색 (카메라 메타데이터): " + win.sel.top + " / " + win.sel.bottom : ""
                               color: c.sub; font.pixelSize: 12; wrapMode: Text.Wrap; width: parent.width }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: "black"; radius: 3; border.color: c.border
                    VideoOutput { id: vout; anchors.fill: parent; anchors.margins: 1 }
                    Text { visible: !win.sel; anchors.centerIn: parent; text: "영상"; color: c.sub }
                }
                RowLayout {
                    Layout.fillWidth: true; spacing: 8
                    Rectangle {
                        width: 32; height: 26; radius: 3; color: c.card; border.color: c.border
                        Text { anchors.centerIn: parent; color: c.text
                               text: player.playbackState === MediaPlayer.PlayingState ? "⏸" : "▶" }
                        MouseArea { anchors.fill: parent
                                    onClicked: player.playbackState === MediaPlayer.PlayingState ? player.pause() : player.play() }
                    }
                    Slider { Layout.fillWidth: true; from: 0; to: Math.max(1, player.duration); value: player.position
                             onMoved: player.position = value }
                    Text { text: win.mmss(player.position / 1000); color: c.sub; font.pixelSize: 12 }
                }
            }
        }
    }
}
