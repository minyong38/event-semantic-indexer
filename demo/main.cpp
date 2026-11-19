// 검색 데모 클라이언트 (Windows, Qt Quick)
//   search_demo.exe [서버주소]   기본 http://localhost:8765 (WSL의 search_server)
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    const QString server = argc > 1 ? QString::fromLocal8Bit(argv[1]) : QStringLiteral("http://localhost:8765");
    engine.rootContext()->setContextProperty("serverUrl", server);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
                     [] { QCoreApplication::exit(-1); }, Qt::QueuedConnection);
    engine.loadFromModule("SearchDemo", "Main");
    return app.exec();
}
