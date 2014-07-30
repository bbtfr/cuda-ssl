#!/usr/bin/env ruby

Rules = {
  /\( (?=\S)/ => "(\\1",
  /(?<=\S) \)/ => "\\1)",
  /(if|for|switch)\(/ => "\\1 (",
  /POLARSSL_/ => "CUDASSL_",
  /polarssl_/ => "",
}

Dir["**/*.cu", "**/*.c", "**/*.h"].each do |file|

  File.open(file) do |f|
    text = f.read
    origin = text.dup
    
    Rules.each do |from, to|
      text.gsub!(from, to)
    end

    if origin != text
      text.split("\n").zip origin.split("\n") do |t, o|
        puts t, o, nil if t != o
      end

      File.open("#{file}.bak", "w") do |f|
        f.puts origin
      end unless File.exist?("#{file}.bak")

      File.open(file, "w") do |f|
        f.puts text
      end
    end
  end

end