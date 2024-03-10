'use strict'


const TableChart = ({ model, probs }) => {
    return (
        <div>
            <h4>{model}</h4>
            <table className="chart" style={{ width: "100%" }}>
                <tbody>
                    {probs.map(probability =>
                        <tr className="chart-row" key={probability.class + probability.prob}>
                            <td className="chart-label">
                                <span>{probability.class}</span>
                            </td>
                            <td className="chart-bar-table-">
                                <span className="bar-wrapper">
                                    <div className="chart-bar"
                                        style={{ width: `${Math.ceil(probability.prob * 100)}%` }}>
                                    </div>
                                    <span style={{ color: "var(--bg)" }}>x</span>
                                    <span className="chart-percent">
                                        {`${(Math.floor(probability.prob * 1000) / 10)}%`}
                                    </span>
                                </span>
                            </td>
                        </tr>
                    )}
                </tbody>

            </table>
        </div>
    )
}

const ResultStat = ({ timeSpent, topFullLabel }) => {
    const scaleFactors = {
        microsecond: 1000,
        microseconds: 1000,
        millisecond: 1,
        milliseconds: 1,
        second: 0.001,
        seconds: 0.001,
    }

    const scaledTime = timeSpent.span / scaleFactors[timeSpent.unit]
    const displayTime = Math.ceil(scaledTime * 100) / 100

    return (
        <div>
            <div>
                Time spent: <strong>{displayTime} ms</strong>
            </div>
            <div>
                Top full label: <strong>'{topFullLabel}'</strong>
            </div>
        </div>
    )
}

const Charts = ({ results }) => {
    return (
        <div>
            {results.map((res, i) =>
                <div key={res.model} >
                    <TableChart
                        model={res["model"]}
                        probs={res["probs"]}
                    />
                    <ResultStat
                        timeSpent={res["time_spent"]}
                        topFullLabel={res["top_full_label"]}
                    />
                    {i < results.length - 1 ? <hr /> : null}
                </div>
            )}
        </div>
    )
}