'use strict';

const useStatement = (statementInfo) => {
    const [status, setStatus] = useState(undefined)
    const [statement, setStatement] = useState("[No statement]")
    const [results, setResults] = useState([])

    const execute = () => {
        const query = new URLSearchParams({
            name: statementInfo.name
        })

        fetch({
            url: "http://localhost:8080/statements?" + query,
            method: "GET",
        })
        .then(response => {
            if (!response.ok) {
                setStatus(response.status)
                setResults([])
                return
            }

            setStatus(200)
            response.json.then(data => {
                setStatement(data.statement)
                setResults(data.results)
            })
        })
    }

    return {
        status,
        statement,
        results,
        execute,
    }
}

const Statement = (statementInfo) => {
    const statement = useStatement(statementInfo)

    return (
        <article>
            <span>{statement.status}</span>
            <span>{statement.statement}</span>
            <span>{statement.results.size()}</span>
            <button onClick={statement.execute}>Execute</button>
        </article>
    )
}