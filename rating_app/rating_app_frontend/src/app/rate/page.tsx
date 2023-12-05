"use client";

import SpotifyEmbed from "../../components/SpotifyEmbed"
import { useState } from "react"

export default function Home() {
  const [rating, setRating] = useState(0)
  const [song, setSong] = useState("59fQzvSv0wPQorW2Bh0pns")

  const handleRating = (event: any) => {
    setRating(event.target.value)
  }

  const handleSong = (event: any) => {
    setSong(event.target.value)
  }

  return (
    <main className="flex flex-col items-center justify-center py-5">
      <div className="flex flex-col items-center justify-center">
        <SpotifyEmbed trackID={song} height={80} />
      </div>
      <div className="flex flex-col items-center justify-center py-5">
        <span className="text-2xl font-bold text-center">How do you feel about this song?</span>

        <div className="grid grid-rows-5 gap-3 wrap mt-3">
          <button className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-4 rounded" value="1" onClick={handleRating}>hate</button>
          <button className="bg-orange-500 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded" value="2" onClick={handleRating}>dislike</button>
          <button className="bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded" value="3" onClick={handleRating}>neutral</button>
          <button className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded" value="4" onClick={handleRating}>like</button>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" value="5" onClick={handleRating}>love</button>
        </div>
      </div>
    </main>
  )
}
