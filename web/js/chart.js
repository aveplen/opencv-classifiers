'use strict'

const Chart = ({ result }) => {
    return (
        <div>
            <h4>{result.model}</h4>

            <ul className="chart">
                <li style={{ background: "skyblue", gridColumnEnd: "span 100" }}>
                    <div></div>
                    <span>100%</span>
                </li>

                {result.probs.map(probability =>
                    <li style={{ gridColumnEnd: `span ${Math.ceil(probability.prob * 100)}` }}>
                        {probability.class.slice(10, 21) + ":"}
                        <span>{`${(Math.floor(probability.prob * 1000) / 10)}% `}</span>
                    </li>
                )}
            </ul>

            <hr />

            <h4>{result.model}</h4>

            <ul className="chart">
                <li style={{ background: "skyblue", gridColumnEnd: "span 100" }}>
                    reference:
                    <span>100%</span>
                </li>

                {result.probs.map(probability =>
                    <li style={{ gridColumnEnd: `span ${Math.ceil(probability.prob * 100)}` }} key={probability.prob}>
                        {probability.class.slice(0, 21) + ":"}
                        <span>{`${(Math.floor(probability.prob * 1000) / 10)}% `}</span>
                    </li>
                )}
            </ul>
        </div>
    )
}

const Charts = ({ results }) => {
    return (
        <div>
            {results.map(res =>
                <Chart result={res} key={res.model} />
            )}
        </div>
    )
}